/*
 * Multinomial Logistic Regression (RLM) Inference Benchmark for Raspberry Pi Pico
 *
 * This program evaluates a multinomial logistic regression (RLM) model for
 * Fashion-MNIST image classification on a Raspberry Pi Pico using the pico-sdk.
 *
 * The model parameters (weights and biases) are stored in flash memory and
 * inference is executed directly on the microcontroller using a fixed
 * evaluation dataset of 300 images.
 *
 * The experimental protocol follows the methodology used in the research:
 *
 *  • Accuracy evaluation on the full test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes for latency estimation
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * This implementation uses a simple float32 ("naive") computation approach
 * and includes a lookup table (LUT) for fast conversion of uint8 pixel values
 * into normalized float values in the range [0,1].
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "model_fashion_rlm.h"
#include "test_fashion_300.h"

// ----------------------------
// CONFIGURATION (experimental methodology)
// ----------------------------
#define WARMUP_PASSES    5
#define MEASURED_PASSES  30
#define N_TEST           FASHION_TEST300_N

static const char *TAG = "FASHION_RLM_NAIVE_PICO_OPT";

// Prevents the compiler from removing loops during optimization (-O3)
static volatile float g_sink = 0.0f;

// Lookup table for uint8 → float normalization [0..1]
static float u8_to_f01[256];

static void init_u8_lut(void) {
    const float inv255 = 1.0f / 255.0f;
    for (int i = 0; i < 256; i++) u8_to_f01[i] = (float)i * inv255;
}

static inline void softmax10(const float z[10], float p[10]) {
    float m = z[0];
    for (int i = 1; i < 10; i++) if (z[i] > m) m = z[i];

    float s = 0.0f;
    for (int i = 0; i < 10; i++) { p[i] = expf(z[i] - m); s += p[i]; }
    const float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < 10; i++) p[i] *= inv;
}

static inline int argmax10(const float p[10]) {
    int im = 0; float vm = p[0];
    for (int i = 1; i < 10; i++) if (p[i] > vm) { vm = p[i]; im = i; }
    return im;
}

static void rlm_infer_u8_opt(const uint8_t x_u8[784], float probs_out[10]) {
    float z[10];

    for (int k = 0; k < 10; k++) {
        const float *wk = &fashion_rlm_W[k][0];
        float acc = fashion_rlm_b[k];
        for (int i = 0; i < 784; i++) {
            acc += wk[i] * u8_to_f01[x_u8[i]];
        }
        z[k] = acc;
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

    // The USB CDC interface sometimes takes a moment to initialize
    sleep_ms(3000);

    init_u8_lut();

    printf("%s | test=%d | warmup=%d | measured=%d\n",
           TAG, N_TEST, WARMUP_PASSES, MEASURED_PASSES);

    // -------------------------
    // ACCURACY + 95% CI (single pass)
    // -------------------------
    int correct = 0;
    for (int i = 0; i < N_TEST; i++) {
        float p10[10];
        rlm_infer_u8_opt(fashion_test300_x[i], p10);
        int pred = argmax10(p10);
        if (pred == (int)fashion_test300_y[i]) correct++;
        g_sink += p10[0];
    }

    const float p = (float)correct / (float)N_TEST;

    // 95% confidence interval (normal approximation of binomial distribution)
    const float se   = sqrtf((p * (1.0f - p)) / (float)N_TEST);
    const float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n", 100.0f * p, correct, N_TEST);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n", 100.0f * lo, 100.0f * hi);

    // -------------------------
    // Warm-up phase
    // -------------------------
    float tmp[10];
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < N_TEST; i++) {
            rlm_infer_u8_opt(fashion_test300_x[i], tmp);
            g_sink += tmp[0];
        }
    }

    // -------------------------
    // Timing measurement
    // -------------------------
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        absolute_time_t t0 = get_absolute_time();
        for (int i = 0; i < N_TEST; i++) {
            rlm_infer_u8_opt(fashion_test300_x[i], tmp);
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
    // Model memory estimation
    // -------------------------
    const size_t mem_params  = sizeof(fashion_rlm_W) + sizeof(fashion_rlm_b);
    const size_t mem_buffers = (10 * sizeof(float)) + (10 * sizeof(float)); // z + probs
    const size_t mem_total_model = mem_params + mem_buffers;

    printf("Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)\n",
           (unsigned)mem_params,  mem_params  / 1024.0f,
           (unsigned)mem_buffers, mem_buffers / 1024.0f,
           (unsigned)mem_total_model, mem_total_model / 1024.0f);

    printf("sink=%.6f\n", (double)g_sink);

    while (true) { tight_loop_contents(); }
}
