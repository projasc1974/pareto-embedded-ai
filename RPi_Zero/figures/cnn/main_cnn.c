/*
 * Convolutional Neural Network (CNN) Inference Benchmark for Raspberry Pi Zero
 *
 * This program evaluates a Convolutional Neural Network (CNN) model for
 * geometric figure classification on a Raspberry Pi Zero.
 *
 * The CNN model parameters (weights and biases) are stored in memory and
 * inference is executed directly on the CPU using a fixed embedded dataset.
 *
 * The evaluation protocol follows the experimental methodology used in the study:
 *
 *  • Accuracy evaluation on the full test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes for latency estimation
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * The implementation uses a naive float32 approach to ensure fair comparison
 * with other embedded platforms such as ESP32-S3 and Raspberry Pi Pico.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "cnn_weights.h"
#include "test_embedded.h"

#define TAG "FIGURES_CNN_ZERO"

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

volatile float sink = 0.0f;

static inline float relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

static void softmaxN(const float *z, float *p, int n)
{
    float m = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > m) m = z[i];
    }

    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        p[i] = expf(z[i] - m);
        s += p[i];
    }

    float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < n; i++) {
        p[i] *= inv;
    }
}

static int argmaxN(const float *p, int n)
{
    int k = 0;
    float m = p[0];

    for (int i = 1; i < n; i++) {
        if (p[i] > m) {
            m = p[i];
            k = i;
        }
    }

    return k;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    return (double)t.tv_sec * 1000.0 +
           (double)t.tv_nsec / 1e6;
}

static void mean_std(const double *x, int n, double *mean, double *std)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        m += x[i];
    }
    m /= (double)n;

    double v = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - m;
        v += d * d;
    }
    v /= (double)n;

    *mean = m;
    *std = sqrt(v);
}

/*
  Naive CNN architecture:

    Conv(valid 3x3, 8 filters) -> ReLU
    MaxPool(2x2, stride=2) -> 13x13x8
    Flatten (NHWC) -> Dense(32) -> ReLU -> Dense(3) -> Softmax

  IMPORTANT:
  - Correct class order for the Figures CNN model:
      0 = circle
      1 = triangle
      2 = square
  - Do NOT apply class remapping 1 <-> 2
  - Correct normalization: x / 255.0
*/

static void cnn_infer_u8(const uint8_t x_u8[IMAGE_SIZE], float probs[CNN_OUTPUT_SIZE])
{
    static float conv_out[26 * 26 * CNN_FILTERS];
    static float pool_out[CNN_FLATTEN_SIZE];

    const float inv255 = 1.0f / 255.0f;

    // Convolution + ReLU
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

                conv_out[((r * 26 + c) * CNN_FILTERS) + f] = relu(acc);
            }
        }
    }

    // MaxPool 2x2 stride 2
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

    // Dense hidden layer
    float h[CNN_HIDDEN_SIZE];
    for (int o = 0; o < CNN_HIDDEN_SIZE; o++) {
        float acc = cnn_dense_biases[o];

        for (int i = 0; i < CNN_FLATTEN_SIZE; i++) {
            acc += cnn_dense_weights[i][o] * pool_out[i];
        }

        h[o] = relu(acc);
    }

    // Output dense layer
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

int main(void)
{
    printf("%s: test=%d | warmup=%d | measured=%d\n",
           TAG, TEST_IMAGES_COUNT, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95 --------
    int correct = 0;

    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        float p[CNN_OUTPUT_SIZE];
        cnn_infer_u8(test_images[i], p);

        int pred = argmaxN(p, CNN_OUTPUT_SIZE);

        // No class swap for CNN Figures model
        if (pred == (int)test_labels[i]) {
            correct++;
        }

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

        printf("Warmup %d/%d ready\n", w + 1, WARMUP_PASSES);
    }

    // -------- Benchmark --------
    double ms[MEASURED_PASSES];

    for (int k = 0; k < MEASURED_PASSES; k++) {
        double t0 = now_ms();

        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            float p[CNN_OUTPUT_SIZE];
            cnn_infer_u8(test_images[i], p);
            sink += p[2 % CNN_OUTPUT_SIZE];
        }

        double t1 = now_ms();
        ms[k] = t1 - t0;

        printf("Pass %d/%d ready (%.2f ms)\n",
               k + 1, MEASURED_PASSES, ms[k]);
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double latency = mean_ms / (double)TEST_IMAGES_COUNT;
    double throughput = 1000.0 / latency;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n",
           latency, throughput);

    // -------- Model memory estimation --------
    size_t params_bytes =
        sizeof(cnn_conv_weights) + sizeof(cnn_conv_biases) +
        sizeof(cnn_dense_weights) + sizeof(cnn_dense_biases) +
        sizeof(cnn_output_weights) + sizeof(cnn_output_biases);

    size_t buffers_bytes =
        (size_t)(26 * 26 * CNN_FILTERS +
                 CNN_FLATTEN_SIZE +
                 CNN_HIDDEN_SIZE +
                 CNN_OUTPUT_SIZE +
                 CNN_OUTPUT_SIZE) * sizeof(float);

    size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%zu bytes (%.2f KB), buffers=%zu bytes (%.2f KB), total=%zu bytes (%.2f KB)\n",
           params_bytes, params_bytes / 1024.0,
           buffers_bytes, buffers_bytes / 1024.0,
           total_bytes, total_bytes / 1024.0);

    printf("sink=%f\n", sink);

    return 0;
}
