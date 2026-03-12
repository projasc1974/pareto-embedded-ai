/*
 * Multilayer Perceptron (MLP) Inference Benchmark for Raspberry Pi Pico
 *
 * This program evaluates a Multilayer Perceptron (MLP) model for
 * Fashion-MNIST image classification on a Raspberry Pi Pico using the
 * pico-sdk development environment.
 *
 * The model parameters (weights and biases) are embedded in flash memory.
 * Inference is executed directly on the microcontroller using a fixed
 * evaluation dataset of 300 images.
 *
 * The experimental protocol follows the methodology used in the study:
 *
 *  • Accuracy evaluation on the full test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes for latency estimation
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * A lookup table (LUT) is used to convert uint8 pixel values to normalized
 * float values in the range [0,1], improving consistency and performance.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "model_fashion_mlp.h"
#include "test_fashion_300.h"

// ----------------------------
// CONFIGURATION (experimental methodology)
// ----------------------------
#define WARMUP_PASSES    5
#define MEASURED_PASSES  30
#define N_TEST           FASHION_TEST300_N

static const char *TAG = "FASHION_MLP_NAIVE_PICO";

// Anti-optimization (prevents the compiler from removing loops)
static volatile float g_sink = 0.0f;

// LUT u8 -> float [0..1] (faster and consistent normalization)
static float u8_to_f01[256];
static void init_u8_lut(void) {
    const float inv255 = 1.0f / 255.0f;
    for (int i = 0; i < 256; i++) u8_to_f01[i] = (float)i * inv255;
}

static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

static void softmax10(const float z[10], float p[10]) {
    float m = z[0];
    for (int i = 1; i < 10; i++) if (z[i] > m) m = z[i];

    float s = 0.0f;
    for (int i = 0; i < 10; i++) { p[i] = expf(z[i] - m); s += p[i]; }
    const float inv = 1.0f / (s + 1e-12f);
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

/*
  SAME inference procedure used in the ESP32-S3 implementation (main_mlp.c):

  - Input x_u8[784] normalized by /255
  - Dense1: fashion_mlp_w1 (flattened) + fashion_mlp_b1, followed by ReLU
  - Dense2: fashion_mlp_w2 (flattened) + fashion_mlp_b2
  - Softmax output layer with 10 classes
*/
static void mlp_infer_u8(const uint8_t x_u8[784], float probs_out[10]) {
    float h[FASHION_MLP_HIDDEN];
    float z[10];

    // Dense1 + ReLU
    for (int o = 0; o < FASHION_MLP_HIDDEN; o++) {
        float acc = fashion_mlp_b1[o];
        const float *w = &fashion_mlp_w1[o * 784];
        for (int i = 0; i < 784; i++) {
            acc += w[i] * u8_to_f01[x_u8[i]];
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

int main() {
    stdio_init_all();
    sleep_ms(3000);   // USB CDC interface may take some time to initialize
    init_u8_lut();

    printf("%s | test=%d | warmup=%d | measured=%d | hidden=%d\n",
           TAG, N_TEST, WARMUP_PASSES, MEASURED_PASSES, (int)FASHION_MLP_HIDDEN);

    // -------------------------
    // ACCURACY + CI95 (test300)
    // -------------------------
    int correct = 0;
    for (int i = 0; i < N_TEST; i++) {
        float p10[10];
        mlp_infer_u8(fashion_test300_x[i], p10);
        int pred = argmax10(p10);
        if (pred == (int)fashion_test300_y[i]) correct++;
        g_sink += p10[0];
    }

    const float p = (float)correct / (float)N_TEST;

    // CI95 using normal approximation for binomial distribution
    const float se   = sqrtf((p * (1.0f - p)) / (float)N_TEST);
    const float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n", 100.0f * p, correct, N_TEST);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n", 100.0f * lo, 100.0f * hi);

    // -------------------------
    // Warm-up (discarded runs)
    // -------------------------
    float tmp[10];
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < N_TEST; i++) {
            mlp_infer_u8(fashion_test300_x[i], tmp);
            g_sink += tmp[0];
        }
    }

    // -------------------------
    // Timing (30 passes)
    // -------------------------
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        absolute_time_t t0 = get_absolute_time();

        for (int i = 0; i < N_TEST; i++) {
            mlp_infer_u8(fashion_test300_x[i], tmp);
            g_sink += tmp[0];
        }

        absolute_time_t t1 = get_absolute_time();
        int64_t us = absolute_time_diff_us(t0, t1);
        ms[k] = (double)us / 1000.0;
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    const double lat_ms_img = mean_ms / (double)N_TEST;
    const double thr_ips = 1000.0 / lat_ms_img;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n", lat_ms_img, thr_ips);

    // -------------------------
    // Model memory estimation (parameters + buffers)
    // -------------------------
    const size_t mem_params =
        sizeof(fashion_mlp_w1) + sizeof(fashion_mlp_b1) +
        sizeof(fashion_mlp_w2) + sizeof(fashion_mlp_b2);

    // buffers: h[H] + z[10] + probs[10]
    const size_t mem_buffers =
        (size_t)FASHION_MLP_HIDDEN * sizeof(float) +
        10 * sizeof(float) +
        10 * sizeof(float);

    const size_t mem_total = mem_params + mem_buffers;

    printf("Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)\n",
           (unsigned)mem_params,  mem_params  / 1024.0f,
           (unsigned)mem_buffers, mem_buffers / 1024.0f,
           (unsigned)mem_total,   mem_total   / 1024.0f);

    printf("sink=%.6f\n", (double)g_sink);

    while (true) { tight_loop_contents(); }
}
