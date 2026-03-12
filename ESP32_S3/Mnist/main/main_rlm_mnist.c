/*
 * RLM (Multinomial Logistic Regression) Inference Benchmark for ESP32
 *
 * This program evaluates a Multinomial Logistic Regression (RLM) model
 * for handwritten digit classification using the MNIST dataset.
 *
 * The model parameters (weights and biases) are stored in flash memory,
 * and inference is performed directly on a fixed evaluation set of
 * 300 MNIST test images.
 *
 * The experiment follows the methodology used in the paper:
 *
 *  • One deterministic accuracy pass on the evaluation set
 *  • Computation of a 95% confidence interval assuming binomial distribution
 *  • Warm-up phase to stabilize execution
 *  • 30 measured inference passes
 *  • Measurement of latency, throughput and memory usage
 *
 * The implementation intentionally uses a simple ("naive") approach
 * without additional optimizations, in order to maintain consistency
 * with the experimental methodology used for cross-platform comparison.
 */

#include <stdio.h>
#include <math.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "mnist_test300.h"
#include "model_mnist_rlm.h"

#define INPUT_DIM   784
#define OUTPUT_DIM  10

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

static const char *TAG = "MNIST_RLM_NAIVE";

static inline void softmax10(const float z[10], float p[10]) {
    float m = z[0];
    for (int i = 1; i < 10; i++) if (z[i] > m) m = z[i];

    float s = 0.0f;
    for (int i = 0; i < 10; i++) {
        p[i] = expf(z[i] - m);
        s += p[i];
    }
    float inv = 1.0f / (s + 1e-9f);
    for (int i = 0; i < 10; i++) p[i] *= inv;
}

static inline int argmax10(const float p[10]) {
    int im = 0;
    float vm = p[0];
    for (int i = 1; i < 10; i++) {
        if (p[i] > vm) { vm = p[i]; im = i; }
    }
    return im;
}

static void infer_one(const unsigned char *img_u8, float probs_out[10]) {
    float z[10];
    for (int o = 0; o < OUTPUT_DIM; o++) {
        float acc = b[o];
        for (int i = 0; i < INPUT_DIM; i++) {
            float x = ((float)img_u8[i]) / 255.0f; // naive normalization
            acc += W[o][i] * x;
        }
        z[o] = acc;
    }
    softmax10(z, probs_out);
}

static void run_experiment(void) {
    // ====== Marker to confirm a new binary execution ======
    printf("\n=== RUN: MNIST_RLM_NAIVE (with ACCURACY) ===\n");
    fflush(stdout);

    // Heap baseline measurement
    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t min_before  = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

    // ====== Accuracy (single deterministic pass) ======
    int correct = 0;
    for (int i = 0; i < MNIST_TEST_COUNT; i++) {
        float p10[10];
        infer_one(mnist_test_images[i], p10);
        int pred = argmax10(p10);
        if ((unsigned char)pred == mnist_test_labels[i]) correct++;
    }

    float p = (float)correct / (float)MNIST_TEST_COUNT;
    float acc = 100.0f * p;

    // empirical standard deviation for a fixed dataset = 0
    float acc_std = 0.0f;

    // 95% confidence interval
    float se = sqrtf((p * (1.0f - p)) / (float)MNIST_TEST_COUNT);
    float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    // Forced print (independent from ESP_LOG)
    printf("ACCURACY test%d: %.2f%% (std=%.2f) (%d/%d)\n",
           MNIST_TEST_COUNT, acc, acc_std, correct, MNIST_TEST_COUNT);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n",
           100.0f * lo, 100.0f * hi);
    fflush(stdout);

    // Also print using ESP logging
    ESP_LOGI(TAG, "Accuracy test%d: %.2f%% (std=%.2f) (%d/%d)",
             MNIST_TEST_COUNT, acc, acc_std, correct, MNIST_TEST_COUNT);
    ESP_LOGI(TAG, "Accuracy 95%% CI: [%.2f%%, %.2f%%]",
             100.0f * lo, 100.0f * hi);

    // ====== Warm-up phase (discarded measurements) ======
    for (int k = 0; k < WARMUP_PASSES; k++) {
        for (int i = 0; i < MNIST_TEST_COUNT; i++) {
            float p10[10];
            infer_one(mnist_test_images[i], p10);
        }
    }

    // ====== Measurement phase (30 passes) ======
    double sum_ms = 0.0, sum2_ms = 0.0;

    for (int k = 0; k < MEASURED_PASSES; k++) {
        int64_t t0 = esp_timer_get_time();

        for (int i = 0; i < MNIST_TEST_COUNT; i++) {
            float p10[10];
            infer_one(mnist_test_images[i], p10);
        }

        int64_t t1 = esp_timer_get_time();

        double batch_ms = (double)(t1 - t0) / 1000.0;
        sum_ms  += batch_ms;
        sum2_ms += batch_ms * batch_ms;
    }

    double mean_batch_ms = sum_ms / (double)MEASURED_PASSES;
    double var_batch_ms  = (sum2_ms / (double)MEASURED_PASSES) - mean_batch_ms * mean_batch_ms;
    double std_batch_ms  = (var_batch_ms > 0.0) ? sqrt(var_batch_ms) : 0.0;

    double lat_ms_img = mean_batch_ms / (double)MNIST_TEST_COUNT;
    double ips = 1000.0 / lat_ms_img;

    ESP_LOGI(TAG, "Batch mean: %.3f ms | std: %.3f ms", mean_batch_ms, std_batch_ms);
    ESP_LOGI(TAG, "Latency: %.6f ms/img | Throughput: %.2f img/s", lat_ms_img, ips);

    size_t min_after = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    size_t min_during = (min_after < min_before) ? min_after : min_before;
    size_t peak_heap_used = (free_before > min_during) ? (free_before - min_during) : 0;

    ESP_LOGI(TAG, "Peak heap used (approx): %u bytes (%.2f KB)",
             (unsigned)peak_heap_used, peak_heap_used / 1024.0f);

    size_t mem_params = sizeof(W) + sizeof(b);
    size_t mem_buffers = 10 * sizeof(float);
    ESP_LOGI(TAG, "Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB)",
             (unsigned)mem_params, mem_params / 1024.0f,
             (unsigned)mem_buffers, mem_buffers / 1024.0f);
}

void app_main(void) {
    ESP_LOGI(TAG, "MNIST RLM naive | test=%d | warmup=%d | measured=%d",
             MNIST_TEST_COUNT, WARMUP_PASSES, MEASURED_PASSES);

    run_experiment();
}
