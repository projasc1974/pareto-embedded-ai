/*
 * CNN Inference Benchmark for Raspberry Pi Pico
 *
 * This program evaluates a Convolutional Neural Network (CNN) model for
 * classifying geometric figures using the Raspberry Pi Pico and the pico-sdk.
 *
 * The model parameters (weights and biases) are embedded in flash memory
 * and inference is performed directly on the microcontroller using a fixed
 * evaluation dataset.
 *
 * The experimental protocol follows the methodology used in the paper:
 *
 *  • Accuracy evaluation on the full embedded test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes to estimate latency
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * The implementation intentionally uses a simple float32 ("naive")
 * computation approach without hardware acceleration in order to ensure
 * fair comparison with other embedded platforms such as ESP32-S3 and
 * Raspberry Pi Zero.
 */

// main.c (Raspberry Pi Pico - pico-sdk) | FIGURES CNN naive (CORRECT: NO 1<->2 swap)
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "cnn_weights.h"
#include "test_embedded.h"

#define TAG "FIGURES_CNN_NAIVE_PICO"

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

static volatile float sink = 0.0f;

static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

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

/*
  CNN naive:
    Conv(valid 3x3, 8 filters, IN=1) -> ReLU
    MaxPool(2x2, stride2) -> 13x13x8
    Flatten (NHWC r,c,f) -> 1352
    Dense(32) -> ReLU
    Dense(3)  -> Softmax

  Important: this main assumes that the arrays in the header file have
  the following shapes:

    cnn_conv_weights[8][3][3]
    cnn_conv_biases[8]
    cnn_dense_weights[1352][32]
    cnn_dense_biases[32]
    cnn_output_weights[32][3]
    cnn_output_biases[3]

  which matches the previous implementation used for the geometric figures dataset.
*/
static void cnn_infer_u8(const uint8_t x_u8[IMAGE_SIZE], float probs[CNN_OUTPUT_SIZE]) {
    // Convolution output: 26x26x8 (valid)
    static float conv_out[26 * 26 * CNN_FILTERS];

    // Pool output: 13x13x8 -> flattened size
    static float pool_out[CNN_FLATTEN_SIZE];

    const float inv255 = 0.00392156886f; // 1/255 normalization factor

    // ----- Convolution + ReLU -----
    for (int r = 0; r < 26; r++) {
        for (int c = 0; c < 26; c++) {
            for (int f = 0; f < CNN_FILTERS; f++) {
                float acc = cnn_conv_biases[f];

                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ir = r + kh;
                        int ic = c + kw;
                        float x = (float)x_u8[ir * 28 + ic] * inv255;
                        acc += cnn_conv_weights[f][kh][kw] * x;
                    }
                }

                acc = relu(acc);
                conv_out[((r * 26 + c) * CNN_FILTERS) + f] = acc; // NHWC layout
            }
        }
    }

    // ----- MaxPool 2x2 stride 2 -----
    for (int pr = 0; pr < 13; pr++) {
        for (int pc = 0; pc < 13; pc++) {

            int r0 = pr * 2;
            int c0 = pc * 2;

            for (int f = 0; f < CNN_FILTERS; f++) {

                float a = conv_out[(((r0 + 0) * 26 + (c0 + 0)) * CNN_FILTERS) + f];
                float b = conv_out[(((r0 + 0) * 26 + (c0 + 1)) * CNN_FILTERS) + f];
                float c = conv_out[(((r0 + 1) * 26 + (c0 + 0)) * CNN_FILTERS) + f];
                float d = conv_out[(((r0 + 1) * 26 + (c0 + 1)) * CNN_FILTERS) + f];

                float m = a;
                if (b > m) m = b;
                if (c > m) m = c;
                if (d > m) m = d;

                pool_out[((pr * 13 + pc) * CNN_FILTERS) + f] = m;
            }
        }
    }

    // ----- Dense(32) + ReLU -----
    float h[CNN_HIDDEN_SIZE];

    for (int o = 0; o < CNN_HIDDEN_SIZE; o++) {
        float acc = cnn_dense_biases[o];

        for (int i = 0; i < CNN_FLATTEN_SIZE; i++) {
            acc += cnn_dense_weights[i][o] * pool_out[i];
        }

        h[o] = relu(acc);
    }

    // ----- Dense(3) -----
    float z[CNN_OUTPUT_SIZE];

    for (int o = 0; o < CNN_OUTPUT_SIZE; o++) {
        float acc = cnn_output_biases[o];

        for (int i = 0; i < CNN_HIDDEN_SIZE; i++) {
            acc += cnn_output_weights[i][o] * h[i];
        }

        z[o] = acc;
    }

    softmaxN(z, probs, CNN_OUTPUT_SIZE);
}

int main(void) {

    stdio_init_all();
    sleep_ms(2000);

    printf("%s: test=%d | warmup=%d | measured=%d\n",
           TAG, TEST_IMAGES_COUNT, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95% (single pass) --------
    int correct = 0;

    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {

        float p[CNN_OUTPUT_SIZE];

        cnn_infer_u8(test_images[i], p);

        int pred = argmaxN(p, CNN_OUTPUT_SIZE);

        // IMPORTANT: do NOT apply the 1<->2 swap for this CNN
        // Correct model order: 0=circle, 1=triangle, 2=square

        if (pred == (int)test_labels[i]) correct++;

        sink += p[0];
    }

    float phat = (float)correct / (float)TEST_IMAGES_COUNT;

    float se   = sqrtf((phat * (1.0f - phat)) / (float)TEST_IMAGES_COUNT);
    float ci95 = 1.96f * se;

    float lo = phat - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = phat + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n",
           100.0f * phat, correct, TEST_IMAGES_COUNT);

    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n",
           100.0f * lo, 100.0f * hi);

    // -------- Warm-up --------
    for (int w = 0; w < WARMUP_PASSES; w++) {

        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {

            float p[CNN_OUTPUT_SIZE];

            cnn_infer_u8(test_images[i], p);

            sink += p[1 % CNN_OUTPUT_SIZE];
        }
    }

    // -------- Timing --------
    double ms[MEASURED_PASSES];

    for (int k = 0; k < MEASURED_PASSES; k++) {

        uint64_t t0 = time_us_64();

        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {

            float p[CNN_OUTPUT_SIZE];

            cnn_infer_u8(test_images[i], p);

            sink += p[2 % CNN_OUTPUT_SIZE];
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

    // -------- Model memory estimate (parameters + buffers + total) --------

    const size_t params_bytes =
        sizeof(cnn_conv_weights) + sizeof(cnn_conv_biases) +
        sizeof(cnn_dense_weights) + sizeof(cnn_dense_biases) +
        sizeof(cnn_output_weights) + sizeof(cnn_output_biases);

    const size_t buffers_bytes =
        (size_t)(26 * 26 * CNN_FILTERS +
                 CNN_FLATTEN_SIZE +
                 CNN_HIDDEN_SIZE +
                 CNN_OUTPUT_SIZE +
                 CNN_OUTPUT_SIZE) * sizeof(float);

    const size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)\n",
           (unsigned)params_bytes,  params_bytes / 1024.0f,
           (unsigned)buffers_bytes, buffers_bytes / 1024.0f,
           (unsigned)total_bytes,   total_bytes / 1024.0f);

    printf("sink=%f\n", (double)sink);

    while (true) { tight_loop_contents(); }

    return 0;
}
