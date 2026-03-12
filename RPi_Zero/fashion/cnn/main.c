/*
 * Convolutional Neural Network (CNN) Inference Benchmark for Raspberry Pi Zero
 *
 * This program evaluates a Convolutional Neural Network (CNN) model for
 * Fashion-MNIST image classification on a Raspberry Pi Zero.
 *
 * The neural network parameters (weights and biases) are stored in memory
 * and inference is executed directly on the CPU using a fixed embedded
 * dataset of 300 Fashion-MNIST images.
 *
 * The evaluation protocol follows the experimental methodology used in the study:
 *
 *  • Accuracy evaluation on the full test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes to estimate latency
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * The implementation uses a naive float32 computation approach to ensure
 * fair comparison with other embedded platforms.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "model_fashion_cnn.h"
#include "test_fashion_300.h"

#define TAG "FASHION_CNN_ZERO"
#define WARMUP_PASSES 5
#define MEASURED_PASSES 30

// Prevents compiler optimization from removing loops
volatile float sink = 0.0f;

static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

static void softmax10(const float *z, float *p) {
    float m = z[0];
    for (int i = 1; i < 10; i++) {
        if (z[i] > m) m = z[i];
    }

    float s = 0.0f;
    for (int i = 0; i < 10; i++) {
        p[i] = expf(z[i] - m);
        s += p[i];
    }

    float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < 10; i++) {
        p[i] *= inv;
    }
}

static int argmax10(const float *p) {
    int k = 0;
    float m = p[0];

    for (int i = 1; i < 10; i++) {
        if (p[i] > m) {
            m = p[i];
            k = i;
        }
    }

    return k;
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    return (double)t.tv_sec * 1000.0 +
           (double)t.tv_nsec / 1e6;
}

static void mean_std(const double *x, int n, double *mean, double *std) {
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
  CNN naive architecture:

    Conv(valid 3x3, 8 filters) -> ReLU
    MaxPool(2x2, stride=2) -> 13x13x8
    Flatten (NHWC)
    Dense(32) -> ReLU
    Dense(10) -> Softmax

  Important notes:
  - Fashion-MNIST does not require class remapping
  - Correct normalization: x / 255.0
  - conv_w layout used in this header: [OUT_CH][9]
*/

static void cnn_infer_u8(const uint8_t x_u8[784], float probs[10]) {
    static float conv_out[26 * 26 * FASHION_CNN_CONV_OUT_CH];
    static float pool_out[FASHION_CNN_FLAT];

    const float inv255 = 1.0f / 255.0f;

    // Convolution layer + ReLU
    for (int r = 0; r < 26; r++) {
        for (int c = 0; c < 26; c++) {
            for (int oc = 0; oc < FASHION_CNN_CONV_OUT_CH; oc++) {
                float acc = fashion_cnn_conv_b[oc];

                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ir = r + kh;
                        int ic = c + kw;
                        float x = (float)x_u8[ir * 28 + ic] * inv255;
                        acc += fashion_cnn_conv_w[oc * 9 + (kh * 3 + kw)] * x;
                    }
                }

                conv_out[((r * 26 + c) * FASHION_CNN_CONV_OUT_CH) + oc] = relu(acc);
            }
        }
    }

    // Max pooling layer 2x2 with stride 2
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

    // Dense hidden layer
    float h[FASHION_CNN_D1_UNITS];
    for (int o = 0; o < FASHION_CNN_D1_UNITS; o++) {
        float acc = fashion_cnn_d1_b[o];
        const float *w = &fashion_cnn_d1_w[o * FASHION_CNN_FLAT];

        for (int i = 0; i < FASHION_CNN_FLAT; i++) {
            acc += w[i] * pool_out[i];
        }

        h[o] = relu(acc);
    }

    // Output dense layer
    float z[10];
    for (int o = 0; o < 10; o++) {
        float acc = fashion_cnn_d2_b[o];
        const float *w = &fashion_cnn_d2_w[o * FASHION_CNN_D1_UNITS];

        for (int i = 0; i < FASHION_CNN_D1_UNITS; i++) {
            acc += w[i] * h[i];
        }

        z[o] = acc;
    }

    softmax10(z, probs);
}

int main(void) {
    printf("%s: test=%d | warmup=%d | measured=%d\n",
           TAG, FASHION_TEST300_N, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95 --------
    int correct = 0;

    for (int i = 0; i < FASHION_TEST300_N; i++) {
        float p[10];
        cnn_infer_u8(fashion_test300_x[i], p);

        int pred = argmax10(p);

        if (pred == (int)fashion_test300_y[i]) {
            correct++;
        }

        sink += p[0];
    }

    float phat = (float)correct / (float)FASHION_TEST300_N;
    float se   = sqrtf((phat * (1.0f - phat)) / (float)FASHION_TEST300_N);
    float ci95 = 1.96f * se;

    float lo = phat - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = phat + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n",
           100.0f * phat, correct, FASHION_TEST300_N);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n",
           100.0f * lo, 100.0f * hi);

    // -------- Warm-up phase --------
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < FASHION_TEST300_N; i++) {
            float p[10];
            cnn_infer_u8(fashion_test300_x[i], p);
            sink += p[1];
        }

        printf("Warmup %d/%d listo\n", w + 1, WARMUP_PASSES);
    }

    // -------- Benchmark phase --------
    double ms[MEASURED_PASSES];

    for (int k = 0; k < MEASURED_PASSES; k++) {
        double t0 = now_ms();

        for (int i = 0; i < FASHION_TEST300_N; i++) {
            float p[10];
            cnn_infer_u8(fashion_test300_x[i], p);
            sink += p[2];
        }

        double t1 = now_ms();
        ms[k] = t1 - t0;

        printf("Pass %d/%d listo (%.2f ms)\n",
               k + 1, MEASURED_PASSES, ms[k]);
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double latency = mean_ms / (double)FASHION_TEST300_N;
    double throughput = 1000.0 / latency;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n",
           latency, throughput);

    // -------- Model memory estimation --------
    size_t params_bytes =
        sizeof(fashion_cnn_conv_w) + sizeof(fashion_cnn_conv_b) +
        sizeof(fashion_cnn_d1_w)   + sizeof(fashion_cnn_d1_b) +
        sizeof(fashion_cnn_d2_w)   + sizeof(fashion_cnn_d2_b);

    size_t buffers_bytes =
        (size_t)(26 * 26 * FASHION_CNN_CONV_OUT_CH +
                 FASHION_CNN_FLAT +
                 FASHION_CNN_D1_UNITS +
                 10 + 10) * sizeof(float);

    size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%zu bytes (%.2f KB), buffers=%zu bytes (%.2f KB), total=%zu bytes (%.2f KB)\n",
           params_bytes, params_bytes / 1024.0,
           buffers_bytes, buffers_bytes / 1024.0,
           total_bytes, total_bytes / 1024.0);

    printf("sink=%f\n", sink);

    return 0;
}
