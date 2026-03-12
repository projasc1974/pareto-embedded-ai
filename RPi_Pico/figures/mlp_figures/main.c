/*
 * Logistic Regression (RLM) Inference Benchmark for Raspberry Pi Pico
 *
 * This program evaluates a multinomial logistic regression (RLM) model
 * running on a Raspberry Pi Pico using the pico-sdk environment.
 *
 * The model classifies simple geometric figures using an embedded dataset.
 * Model parameters (weights and biases) are stored in flash memory and
 * inference is executed directly on the microcontroller.
 *
 * The experiment follows the evaluation methodology used in the paper:
 *
 *  • Warm-up phase to stabilize runtime behavior
 *  • 30 measured inference passes for latency estimation
 *  • Computation of latency and throughput
 *  • Accuracy evaluation on the full embedded test set
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Estimation of model memory usage
 *
 * The implementation intentionally uses a simple float32 "naive" approach
 * without hardware acceleration in order to ensure fair comparison with
 * other embedded platforms such as ESP32-S3 and Raspberry Pi Zero.
 */

#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "rlm_weights.h"
#include "test_embedded.h"

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

static volatile float sink = 0.0f;

static void softmaxN(const float *z, float *p, int n) {
    float m = z[0];
    for (int i = 1; i < n; i++) if (z[i] > m) m = z[i];

    float s = 0.0f;
    for (int i = 0; i < n; i++) { p[i] = expf(z[i] - m); s += p[i]; }

    float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < n; i++) p[i] *= inv;
}

static int argmaxN(const float *p, int n) {
    int k = 0; float m = p[0];
    for (int i = 1; i < n; i++) if (p[i] > m) { m = p[i]; k = i; }
    return k;
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

// RLM inference: converts uint8 input image to float in range [0..1]
static void rlm_infer_u8(const uint8_t x_u8[IMAGE_SIZE], float probs[RLM_CLASSES]) {
    float z[RLM_CLASSES];

    for (int c = 0; c < RLM_CLASSES; c++) {
        float acc = rlm_biases[c];
        for (int i = 0; i < RLM_FEATURES; i++) {
            float xi = ((float)x_u8[i]) / 255.0f;
            acc += rlm_weights[c][i] * xi;
        }
        z[c] = acc;
    }

    softmaxN(z, probs, RLM_CLASSES);
}

int main(void) {
    stdio_init_all();
    sleep_ms(2000);

    

    // -------------------------
    // 2) Warm-up phase
    // -------------------------
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            float p[RLM_CLASSES];
            rlm_infer_u8(test_images[i], p);
            sink += p[1 % RLM_CLASSES];
        }
    }

    // -------------------------
    // 3) Timing measurements
    // -------------------------
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        uint64_t t0 = time_us_64();

        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            float p[RLM_CLASSES];
            rlm_infer_u8(test_images[i], p);
            sink += p[2 % RLM_CLASSES];
        }

        uint64_t t1 = time_us_64();
        ms[k] = (double)(t1 - t0) / 1000.0;
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)TEST_IMAGES_COUNT;
    double thr_ips = 1000.0 / lat_ms_img;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n", lat_ms_img, thr_ips);
    
    
    // -------------------------
    // 1) Accuracy + CI95 (test300)
    // -------------------------
    int correct = 0;
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        float p[RLM_CLASSES];
        rlm_infer_u8(test_images[i], p);
        int pred = argmaxN(p, RLM_CLASSES);
        
        // FIX: align model indices with labels from test_embedded.h
        // (in the headers: square=2 and triangle=1, but the model uses square=1 triangle=2)
        if (pred == 1) pred = 2;
        else if (pred == 2) pred = 1;
        
        if (pred == (int)test_labels[i]) correct++;
        sink += p[0];
    }

    float phat = (float)correct / (float)TEST_IMAGES_COUNT;

    // 95% confidence interval (normal approximation) for a proportion
    float se   = sqrtf((phat * (1.0f - phat)) / (float)TEST_IMAGES_COUNT);
    float ci95 = 1.96f * se;

    float lo = phat - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = phat + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n",
           100.0f * phat, correct, TEST_IMAGES_COUNT);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n",
           100.0f * lo, 100.0f * hi);

    // -------------------------
    // 4) Model memory usage (parameters + buffers + total)
    // -------------------------
    size_t params_bytes  = sizeof(rlm_weights) + sizeof(rlm_biases);
    size_t buffers_bytes = (size_t)(RLM_CLASSES + RLM_CLASSES) * sizeof(float); // z + probs
    size_t total_bytes   = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)\n",
           (unsigned)params_bytes,  params_bytes / 1024.0f,
           (unsigned)buffers_bytes, buffers_bytes / 1024.0f,
           (unsigned)total_bytes,   total_bytes / 1024.0f);

    printf("sink=%f\n", (double)sink);

    while (true) { tight_loop_contents(); }
    return 0;
}
