/*
 * Convolutional Neural Network (CNN) Inference Benchmark for Raspberry Pi Pico
 *
 * This program evaluates a Convolutional Neural Network (CNN) model for
 * Fashion-MNIST image classification on a Raspberry Pi Pico using the pico-sdk.
 *
 * The CNN model parameters (weights and biases) are stored in flash memory.
 * Inference is executed directly on the microcontroller using a fixed
 * evaluation dataset of 300 images.
 *
 * The experimental protocol follows the methodology used in the research:
 *
 *  • Accuracy evaluation on the complete test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes to estimate latency
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * The implementation uses a simple float32 ("naive") approach without
 * hardware acceleration to allow fair comparison with other embedded
 * platforms such as ESP32-S3 and Raspberry Pi Zero.
 */

// main.c (Raspberry Pi Pico - pico-sdk) | Fashion-MNIST CNN naive (FIXED LAYOUT)
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "model_fashion_cnn.h"
#include "test_fashion_300.h"

#define TAG "FASHION_CNN_NAIVE_PICO"

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

/*
  IMPORTANT FIX #1:
  Conv2D weights from TensorFlow/Keras are typically stored as:
     (KH, KW, IN_CH, OUT_CH) with OUT_CH as the fastest dimension.
  With IN_CH = 1:
     idx = (kh * KW + kw) * OUT_CH + oc
*/
static inline float conv_w_at(int oc, int kh, int kw) {
    int k = kh * FASHION_CNN_CONV_KW + kw; // 0..8
    return fashion_cnn_conv_w[oc * (FASHION_CNN_CONV_KH * FASHION_CNN_CONV_KW) + k];
}

// CNN naive: Conv(valid 3x3, 8) -> ReLU -> MaxPool(2x2) -> Flatten(NHWC) -> Dense32(ReLU) -> Dense10 -> Softmax
static void cnn_infer_u8(const uint8_t x_u8[784], float probs_out[10]) {
    // NHWC layout:
    // conv_out[r][c][oc] => index = ((r*26 + c) * OUT_CH + oc)
    static float conv_out[26 * 26 * FASHION_CNN_CONV_OUT_CH];

    // pool_out[pr][pc][oc] => index = ((pr*13 + pc) * OUT_CH + oc)
    static float pool_out[FASHION_CNN_FLAT];

    // ----- Convolution + ReLU (VALID) -----
    for (int r = 0; r < 26; r++) {
        for (int c = 0; c < 26; c++) {
            for (int oc = 0; oc < FASHION_CNN_CONV_OUT_CH; oc++) {
                float acc = fashion_cnn_conv_b[oc];

                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ir = r + kh;
                        int ic = c + kw;
                        float x = ((float)x_u8[ir * 28 + ic]) / 255.0f;
                        acc += conv_w_at(oc, kh, kw) * x;
                    }
                }

                acc = relu(acc);
                conv_out[((r * 26 + c) * FASHION_CNN_CONV_OUT_CH) + oc] = acc;
            }
        }
    }

    // ----- MaxPool 2x2 (stride 2) -----
    for (int pr = 0; pr < 13; pr++) {
        for (int pc = 0; pc < 13; pc++) {
            int r0 = pr * 2;
            int c0 = pc * 2;

            for (int oc = 0; oc < FASHION_CNN_CONV_OUT_CH; oc++) {
                float a = conv_out[(((r0 + 0) * 26 + (c0 + 0)) * FASHION_CNN_CONV_OUT_CH) + oc];
                float b = conv_out[(((r0 + 0) * 26 + (c0 + 1)) * FASHION_CNN_CONV_OUT_CH) + oc];
                float c = conv_out[(((r0 + 1) * 26 + (c0 + 0)) * FASHION_CNN_CONV_OUT_CH) + oc];
                float d = conv_out[(((r0 + 1) * 26 + (c0 + 1)) * FASHION_CNN_CONV_OUT_CH) + oc];

                float m = a;
                if (b > m) m = b;
                if (c > m) m = c;
                if (d > m) m = d;

                pool_out[((pr * 13 + pc) * FASHION_CNN_CONV_OUT_CH) + oc] = m;
            }
        }
    }

    // ----- Dense1 (32) + ReLU -----
    float h[FASHION_CNN_D1_UNITS];
    for (int o = 0; o < FASHION_CNN_D1_UNITS; o++) {
        float acc = fashion_cnn_d1_b[o];
        const float *w = &fashion_cnn_d1_w[o * FASHION_CNN_FLAT];
        for (int i = 0; i < FASHION_CNN_FLAT; i++) acc += w[i] * pool_out[i];
        h[o] = relu(acc);
    }

    // ----- Dense2 (10) -----
    float z[10];
    for (int o = 0; o < 10; o++) {
        float acc = fashion_cnn_d2_b[o];
        const float *w = &fashion_cnn_d2_w[o * FASHION_CNN_D1_UNITS];
        for (int i = 0; i < FASHION_CNN_D1_UNITS; i++) acc += w[i] * h[i];
        z[o] = acc;
    }

    softmax10(z, probs_out);
}

// Anti-optimization variable
static volatile float sink = 0.0f;

int main(void) {
    stdio_init_all();
    sleep_ms(3000);

    printf("%s: Fashion-MNIST CNN naive | test=%d | warmup=%d | measured=%d\n",
           TAG, FASHION_TEST300_N, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95% --------
    int correct = 0;
    for (int i = 0; i < FASHION_TEST300_N; i++) {
        float p10[10];
        cnn_infer_u8(fashion_test300_x[i], p10);
        int pred = argmax10(p10);
        if (pred == (int)fashion_test300_y[i]) correct++;
        sink += p10[0];
    }

    float p = (float)correct / (float)FASHION_TEST300_N;
    float acc = 100.0f * p;

    float se = sqrtf((p * (1.0f - p)) / (float)FASHION_TEST300_N);
    float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n", acc, correct, FASHION_TEST300_N);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n", 100.0f * lo, 100.0f * hi);

    // -------- Warm-up --------
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < FASHION_TEST300_N; i++) {
            float p10[10];
            cnn_infer_u8(fashion_test300_x[i], p10);
            sink += p10[1];
        }
    }

    // -------- Timing --------
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        uint64_t t0 = time_us_64();
        for (int i = 0; i < FASHION_TEST300_N; i++) {
            float p10[10];
            cnn_infer_u8(fashion_test300_x[i], p10);
            sink += p10[2];
        }
        uint64_t t1 = time_us_64();
        ms[k] = (double)(t1 - t0) / 1000.0;
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)FASHION_TEST300_N;
    double thr_ips = 1000.0 / lat_ms_img;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n", lat_ms_img, thr_ips);

    // -------- Model memory estimation --------
    size_t params_bytes =
        (size_t)(72 + 8 + 43264 + 32 + (10 * FASHION_CNN_D1_UNITS) + 10) * sizeof(float);

    size_t buffers_bytes =
        (size_t)(26 * 26 * FASHION_CNN_CONV_OUT_CH +
                 FASHION_CNN_FLAT +
                 FASHION_CNN_D1_UNITS +
                 10 + 10) * sizeof(float);

    size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)\n",
           (unsigned)params_bytes,  params_bytes / 1024.0f,
           (unsigned)buffers_bytes, buffers_bytes / 1024.0f,
           (unsigned)total_bytes,   total_bytes / 1024.0f);

    printf("sink=%f\n", (double)sink);

    while (true) { tight_loop_contents(); }
    return 0;
}
