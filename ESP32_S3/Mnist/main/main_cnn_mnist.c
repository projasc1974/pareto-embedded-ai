/*
 * CNN Inference Benchmark for ESP32
 *
 * This program evaluates a Convolutional Neural Network (CNN) model
 * for handwritten digit classification using a subset of the MNIST dataset.
 *
 * The model parameters are stored in flash memory and inference is executed
 * directly on a fixed evaluation set of 300 MNIST images.
 *
 * The implementation follows the experimental methodology used in the paper:
 *
 *  • One deterministic accuracy evaluation over the full dataset
 *  • Computation of a 95% confidence interval assuming a binomial distribution
 *  • Warm-up phase to stabilize execution time
 *  • 30 measured inference passes
 *  • Measurement of latency, throughput and memory usage
 *
 * The CNN architecture implemented is:
 *
 *      Input 28×28×1
 *      → Conv 3×3 (valid) with 8 filters → 26×26×8
 *      → MaxPool 2×2 → 13×13×8
 *      → Flatten → 1352
 *      → Dense (32) + ReLU
 *      → Dense (10)
 *      → Softmax
 *
 * The implementation intentionally uses a simple ("naive") float32 approach
 * without hardware acceleration or vectorization to maintain fair comparison
 * across embedded platforms.
 */

#include <stdio.h>
#include <math.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "model_mnist_cnn.h"
#include "test_mnist_300.h"

static const char *TAG = "MNIST_CNN_NAIVE";

#define WARMUP_PASSES   5
#define MEASURE_PASSES  30

// Input: 28x28x1
// Conv valid 3x3 -> 26x26x8
// MaxPool 2x2 -> 13x13x8
// Flatten -> 1352
#define IN_H 28
#define IN_W 28
#define C1_H 26
#define C1_W 26
#define C1_C 8
#define P1_H 13
#define P1_W 13
#define FLAT 1352  // 13*13*8

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

// Flattened layout for mnist_cnn_conv_w: [8][1][3][3] => 8*9
static inline float conv_w(int oc, int kh, int kw) {
    int idx = (oc * 9) + (kh * 3) + kw;
    return mnist_cnn_conv_w[idx];
}

static void cnn_infer(const float *x784, float out_prob[10]) {
    // static buffers (naive float32 implementation)
    static float c1[C1_H * C1_W * C1_C];
    static float p1[P1_H * P1_W * C1_C];
    static float d1[MNIST_CNN_D1_UNITS];
    float z2[10];

    // Convolution + ReLU
    for (int oc = 0; oc < C1_C; oc++) {
        for (int r = 0; r < C1_H; r++) {
            for (int c = 0; c < C1_W; c++) {
                float acc = mnist_cnn_conv_b[oc];
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        float xv = x784[(r + kh) * IN_W + (c + kw)];
                        acc += xv * conv_w(oc, kh, kw);
                    }
                }
                c1[(oc * C1_H + r) * C1_W + c] = relu(acc);
            }
        }
    }

    // MaxPool 2x2
    for (int oc = 0; oc < C1_C; oc++) {
        for (int r = 0; r < P1_H; r++) {
            for (int c = 0; c < P1_W; c++) {
                int r0 = 2 * r;
                int c0 = 2 * c;
                float m = c1[(oc * C1_H + r0) * C1_W + c0];
                float v1 = c1[(oc * C1_H + r0) * C1_W + (c0 + 1)];
                float v2 = c1[(oc * C1_H + (r0 + 1)) * C1_W + c0];
                float v3 = c1[(oc * C1_H + (r0 + 1)) * C1_W + (c0 + 1)];
                if (v1 > m) m = v1;
                if (v2 > m) m = v2;
                if (v3 > m) m = v3;
                p1[(oc * P1_H + r) * P1_W + c] = m;
            }
        }
    }

    // Dense1 (32) + ReLU | d1_w layout [32][1352] flattened
    // Build flat vector using Keras order: (r, c, oc)
	static float flat[FLAT];
	int t = 0;
	for (int r = 0; r < P1_H; r++) {
		for (int c = 0; c < P1_W; c++) {
			for (int oc = 0; oc < C1_C; oc++) {
				flat[t++] = p1[(oc * P1_H + r) * P1_W + c];  // read using current layout
			}
		}
	}

// Dense1 (32) + ReLU using the correct flattened vector
for (int o = 0; o < MNIST_CNN_D1_UNITS; o++) {
    float acc = mnist_cnn_d1_b[o];
    const float *w = &mnist_cnn_d1_w[o * FLAT];
    for (int i = 0; i < FLAT; i++) acc += flat[i] * w[i];
    d1[o] = relu(acc);
}

    // Dense2 (10) | d2_w layout [10][32] flattened
    for (int o = 0; o < 10; o++) {
        float acc = mnist_cnn_d2_b[o];
        const float *w = &mnist_cnn_d2_w[o * MNIST_CNN_D1_UNITS];
        for (int i = 0; i < MNIST_CNN_D1_UNITS; i++) acc += d1[i] * w[i];
        z2[o] = acc;
    }

    softmax10(z2, out_prob);
}

static void mean_std(const double *x, int n, double *mean, double *std) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= (double)n;
    double v = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - m; v += d * d; }
    v /= (double)n;
    *mean = m;
    *std = sqrt(v);
}

void app_main(void) {
    ESP_LOGI(TAG, "MNIST CNN naive | test=%d | warmup=%d | measured=%d",
             MNIST_TEST300_N, WARMUP_PASSES, MEASURE_PASSES);

    // Heap baseline measurement
    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t min_before  = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

    // =========================
    // ACCURACY test300 (single pass)
    // =========================
    int correct = 0;
    for (int i = 0; i < MNIST_TEST300_N; i++) {
        float p10[10];
        cnn_infer(mnist_test300_x[i], p10);
        int pred = argmax10(p10);
        if (pred == (int)mnist_test300_y[i]) correct++;
    }

    float p = (float)correct / (float)MNIST_TEST300_N;
    float acc = 100.0f * p;
    float acc_std = 0.0f; // fixed dataset → deterministic

    // 95% confidence interval (normal approximation)
    float se = sqrtf((p * (1.0f - p)) / (float)MNIST_TEST300_N);
    float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    // Forced print + logging
    printf("ACCURACY test300: %.2f%% (std=%.2f) (%d/%d)\n", acc, acc_std, correct, MNIST_TEST300_N);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n", 100.0f * lo, 100.0f * hi);

    ESP_LOGI(TAG, "ACCURACY test300: %.2f%% (std=%.2f) (%d/%d)", acc, acc_std, correct, MNIST_TEST300_N);
    ESP_LOGI(TAG, "ACCURACY 95%% CI: [%.2f%%, %.2f%%]", 100.0f * lo, 100.0f * hi);

    // =========================
    // Warm-up (discarded)
    // =========================
    float p10[10];
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < MNIST_TEST300_N; i++) cnn_infer(mnist_test300_x[i], p10);
    }

    // =========================
    // Measurement (30 passes)
    // =========================
    double ms[MEASURE_PASSES];
    for (int k = 0; k < MEASURE_PASSES; k++) {
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < MNIST_TEST300_N; i++) cnn_infer(mnist_test300_x[i], p10);
        int64_t t1 = esp_timer_get_time();
        ms[k] = (double)(t1 - t0) / 1000.0;
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURE_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)MNIST_TEST300_N;
    double thr_ips = 1000.0 / lat_ms_img;

    ESP_LOGI(TAG, "Batch mean: %.3f ms | std: %.3f ms", mean_ms, std_ms);
    ESP_LOGI(TAG, "Latency: %.6f ms/img | Throughput: %.2f img/s", lat_ms_img, thr_ips);

    // Approximate heap peak (note: static buffers are not included)
    size_t min_after = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    size_t min_during = (min_after < min_before) ? min_after : min_before;
    size_t peak_used = (free_before > min_during) ? (free_before - min_during) : 0;
    ESP_LOGI(TAG, "Peak heap used (approx): %u bytes (%.2f KB)",
             (unsigned)peak_used, peak_used / 1024.0f);

    // Model memory estimate (parameters + buffers)
    int conv_params = (8 * 1 * 3 * 3) + 8;
    int d1_params   = (MNIST_CNN_D1_UNITS * FLAT) + MNIST_CNN_D1_UNITS;
    int d2_params   = (10 * MNIST_CNN_D1_UNITS) + 10;
    int total_params = conv_params + d1_params + d2_params;
    int params_bytes = total_params * 4;

    int buffers_f = (C1_H * C1_W * C1_C) + (P1_H * P1_W * C1_C) + MNIST_CNN_D1_UNITS + 10 + 10;
    int buffers_bytes = buffers_f * 4;

    ESP_LOGI(TAG, "Model memory estimate: params=%d bytes (%.2f KB), buffers=%d bytes (%.2f KB)",
             params_bytes, params_bytes / 1024.0f, buffers_bytes, buffers_bytes / 1024.0f);
}
