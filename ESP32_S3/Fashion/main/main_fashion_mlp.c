/*
 * MLP Inference Benchmark for ESP32
 *
 * This program evaluates a Multilayer Perceptron (MLP) model for
 * Fashion-MNIST image classification using a fixed evaluation set
 * of 300 test images.
 *
 * The model parameters are stored in flash memory and inference is
 * executed directly on the embedded device.
 *
 * The experiment follows the methodology used in the paper:
 *
 *  • One deterministic accuracy pass over the full dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution time
 *  • 30 measured inference passes
 *  • Measurement of latency, throughput and memory usage
 *
 * The implementation intentionally uses a simple ("naive") float32
 * approach without hardware acceleration to ensure consistent
 * cross-platform comparisons.
 */

#include <stdio.h>
#include <math.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "model_fashion_mlp.h"
#include "test_fashion_300.h"

static const char *TAG = "FASHION_MLP_NAIVE";

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

static void softmax10(const float z[10], float p[10]) {
    float m = z[0];
    for (int i = 1; i < 10; i++) if (z[i] > m) m = z[i];
    float s = 0.0f;
    for (int i = 0; i < 10; i++) { p[i] = expf(z[i] - m); s += p[i]; }
    float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < 10; i++) p[i] *= inv;
}

static int argmax10(const float p[10]) {
    int k = 0; float m = p[0];
    for (int i = 1; i < 10; i++) if (p[i] > m) { m = p[i]; k = i; }
    return k;
}

// fashion_mlp_w1 layout: [H][784] flattened
static void mlp_infer_u8(const uint8_t x_u8[784], float probs_out[10]) {
    float h[FASHION_MLP_HIDDEN];
    float z[10];

    // Dense1 + ReLU
    for (int o = 0; o < FASHION_MLP_HIDDEN; o++) {
        float acc = fashion_mlp_b1[o];
        const float *w = &fashion_mlp_w1[o * 784];
        for (int i = 0; i < 784; i++) {
            float x = ((float)x_u8[i]) / 255.0f;
            acc += w[i] * x;
        }
        h[o] = relu(acc);
    }

    // Dense2
    for (int o = 0; o < 10; o++) {
        float acc = fashion_mlp_b2[o];
        const float *w = &fashion_mlp_w2[o * FASHION_MLP_HIDDEN];
        for (int i = 0; i < FASHION_MLP_HIDDEN; i++) {
            acc += w[i] * h[i];
        }
        z[o] = acc;
    }

    softmax10(z, probs_out);
}

static void mean_std(const double *x, int n, double *mean, double *std) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= (double)n;
    double v = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - m; v += d * d; }
    v /= (double)n;
    *mean = m;
    *std  = sqrt(v);
}

void app_main(void) {
    ESP_LOGI(TAG, "Fashion-MNIST MLP naive | test=%d | warmup=%d | measured=%d",
             FASHION_TEST300_N, WARMUP_PASSES, MEASURED_PASSES);

    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t min_before  = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

    // Accuracy (single pass)
    int correct = 0;
    for (int i = 0; i < FASHION_TEST300_N; i++) {
        float p10[10];
        mlp_infer_u8(fashion_test300_x[i], p10);
        int pred = argmax10(p10);
        if (pred == (int)fashion_test300_y[i]) correct++;
    }

    float p = (float)correct / (float)FASHION_TEST300_N;
    float acc = 100.0f * p;

    float se = sqrtf((p * (1.0f - p)) / (float)FASHION_TEST300_N);
    float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n", acc, correct, FASHION_TEST300_N);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n", 100.0f * lo, 100.0f * hi);

    ESP_LOGI(TAG, "ACCURACY test300: %.2f%% (std=0.00) (%d/%d)", acc, correct, FASHION_TEST300_N);
    ESP_LOGI(TAG, "ACCURACY 95%% CI: [%.2f%%, %.2f%%]", 100.0f * lo, 100.0f * hi);

    // Warm-up phase
    float p10[10];
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < FASHION_TEST300_N; i++) mlp_infer_u8(fashion_test300_x[i], p10);
    }

    // Timing measurements
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < FASHION_TEST300_N; i++) mlp_infer_u8(fashion_test300_x[i], p10);
        int64_t t1 = esp_timer_get_time();
        ms[k] = (double)(t1 - t0) / 1000.0;
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)FASHION_TEST300_N;
    double thr_ips = 1000.0 / lat_ms_img;

    ESP_LOGI(TAG, "Batch mean: %.3f ms | std: %.3f ms", mean_ms, std_ms);
    ESP_LOGI(TAG, "Latency: %.6f ms/img | Throughput: %.2f img/s", lat_ms_img, thr_ips);

    // Approximate heap peak usage
    size_t min_after = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    size_t min_during = (min_after < min_before) ? min_after : min_before;
    size_t peak_used = (free_before > min_during) ? (free_before - min_during) : 0;
    ESP_LOGI(TAG, "Peak heap used (approx): %u bytes (%.2f KB)",
             (unsigned)peak_used, peak_used / 1024.0f);

    // Model memory estimate
    // params: w1(H*784) + b1(H) + w2(10*H) + b2(10) float32
    size_t params_bytes =
        (size_t)(FASHION_MLP_HIDDEN * 784 + FASHION_MLP_HIDDEN + 10 * FASHION_MLP_HIDDEN + 10) * sizeof(float);

    // buffers: h[H] + z[10] + p[10] float32
    size_t buffers_bytes =
        (size_t)(FASHION_MLP_HIDDEN + 10 + 10) * sizeof(float);

    ESP_LOGI(TAG, "Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB)",
             (unsigned)params_bytes, params_bytes / 1024.0f,
             (unsigned)buffers_bytes, buffers_bytes / 1024.0f);
}
