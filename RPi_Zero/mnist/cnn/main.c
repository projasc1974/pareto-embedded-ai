/*
 * Convolutional Neural Network (CNN) Inference Benchmark for Raspberry Pi Zero
 *
 * This program evaluates a Convolutional Neural Network (CNN) model for
 * MNIST handwritten digit classification using a Raspberry Pi Zero.
 *
 * The CNN parameters (weights and biases) are stored in memory and
 * inference is executed directly on the CPU using a fixed embedded dataset
 * of 300 MNIST test images.
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
 * The implementation uses a naive float32 approach to maintain
 * comparability with other embedded platforms such as ESP32-S3
 * and Raspberry Pi Pico.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "model_mnist_cnn.h"
#include "mnist_test300.h"

#define TAG "MNIST_CNN_ZERO"
#define WARMUP_PASSES 5
#define MEASURED_PASSES 30

// Prevents compiler optimizations that could remove loops
volatile float sink = 0.0f;

static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

static void softmax10(const float *z, float *p) {
    float m = z[0];
    for (int i = 1; i < 10; i++) if (z[i] > m) m = z[i];

    float s = 0.0f;
    for (int i = 0; i < 10; i++) {
        p[i] = expf(z[i] - m);
        s += p[i];
    }

    float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < 10; i++) p[i] *= inv;
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

// Returns current time in milliseconds using a monotonic clock
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

// Computes mean and standard deviation of execution times
static void mean_std(const double *x, int n, double *mean, double *std) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m += x[i];
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

static void cnn_infer_u8(const uint8_t x_u8[784], float probs[10]) {
    static float conv_out[26 * 26 * MNIST_CNN_CONV_OUT_CH];
    static float pool_out[MNIST_CNN_FLAT];
    const float inv255 = 1.0f / 255.0f;

    // Convolution + ReLU
    for (int r = 0; r < 26; r++) {
        for (int c = 0; c < 26; c++) {
            for (int oc = 0; oc < MNIST_CNN_CONV_OUT_CH; oc++) {
                float acc = mnist_cnn_conv_b[oc];

                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ir = r + kh;
                        int ic = c + kw;
                        float x = (float)x_u8[ir * 28 + ic] * inv255;
                        acc += mnist_cnn_conv_w[oc * 9 + (kh * 3 + kw)] * x;
                    }
                }

                conv_out[((r * 26 + c) * MNIST_CNN_CONV_OUT_CH) + oc] = relu(acc);
            }
        }
    }

    // MaxPool 2x2 stride 2
    for (int pr = 0; pr < 13; pr++) {
        for (int pc = 0; pc < 13; pc++) {
            int r0 = pr * 2;
            int c0 = pc * 2;

            for (int oc = 0; oc < MNIST_CNN_CONV_OUT_CH; oc++) {
                float a = conv_out[(((r0 + 0) * 26 + (c0 + 0)) * MNIST_CNN_CONV_OUT_CH) + oc];
                float b = conv_out[(((r0 + 0) * 26 + (c0 + 1)) * MNIST_CNN_CONV_OUT_CH) + oc];
                float c = conv_out[(((r0 + 1) * 26 + (c0 + 0)) * MNIST_CNN_CONV_OUT_CH) + oc];
                float d = conv_out[(((r0 + 1) * 26 + (c0 + 1)) * MNIST_CNN_CONV_OUT_CH) + oc];

                float m = a;
                if (b > m) m = b;
                if (c > m) m = c;
                if (d > m) m = d;

                pool_out[((pr * 13 + pc) * MNIST_CNN_CONV_OUT_CH) + oc] = m;
            }
        }
    }

    // Dense hidden layer
    float h[MNIST_CNN_D1_UNITS];
    for (int o = 0; o < MNIST_CNN_D1_UNITS; o++) {
        float acc = mnist_cnn_d1_b[o];
        const float *w = &mnist_cnn_d1_w[o * MNIST_CNN_FLAT];
        for (int i = 0; i < MNIST_CNN_FLAT; i++) {
            acc += w[i] * pool_out[i];
        }
        h[o] = relu(acc);
    }

    // Output dense layer
    float z[10];
    for (int o = 0; o < 10; o++) {
        float acc = mnist_cnn_d2_b[o];
        const float *w = &mnist_cnn_d2_w[o * MNIST_CNN_D1_UNITS];
        for (int i = 0; i < MNIST_CNN_D1_UNITS; i++) {
            acc += w[i] * h[i];
        }
        z[o] = acc;
    }

    softmax10(z, probs);
}

int main(void) {
    printf("%s: test=%d | warmup=%d | measured=%d\n",
           TAG, MNIST_TEST_COUNT, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95 --------
    int correct = 0;
    for (int i = 0; i < MNIST_TEST_COUNT; i++) {
        float p[10];
        cnn_infer_u8(mnist_test_images[i], p);
        int pred = argmax10(p);

        if (pred == (int)mnist_test_labels[i]) correct++;
        sink += p[0];
    }

    float phat = (float)correct / (float)MNIST_TEST_COUNT;
    float se   = sqrtf((phat * (1.0f - phat)) / (float)MNIST_TEST_COUNT);
    float ci95 = 1.96f * se;

    float lo = phat - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = phat + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n",
           100.0f * phat, correct, MNIST_TEST_COUNT);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n",
           100.0f * lo, 100.0f * hi);

    // -------- Warm-up --------
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < MNIST_TEST_COUNT; i++) {
            float p[10];
            cnn_infer_u8(mnist_test_images[i], p);
            sink += p[1];
        }
        printf("Warmup %d/%d listo\n", w + 1, WARMUP_PASSES);
    }

    // -------- Benchmark --------
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        double t0 = now_ms();

        for (int i = 0; i < MNIST_TEST_COUNT; i++) {
            float p[10];
            cnn_infer_u8(mnist_test_images[i], p);
            sink += p[2];
        }

        double t1 = now_ms();
        ms[k] = t1 - t0;
        printf("Pass %d/%d listo (%.2f ms)\n", k + 1, MEASURED_PASSES, ms[k]);
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double latency = mean_ms / (double)MNIST_TEST_COUNT;
    double throughput = 1000.0 / latency;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n", latency, throughput);

    // -------- Model memory estimation --------
    size_t params_bytes =
        sizeof(mnist_cnn_conv_w) + sizeof(mnist_cnn_conv_b) +
        sizeof(mnist_cnn_d1_w)   + sizeof(mnist_cnn_d1_b) +
        sizeof(mnist_cnn_d2_w)   + sizeof(mnist_cnn_d2_b);

    size_t buffers_bytes =
        (size_t)(26 * 26 * MNIST_CNN_CONV_OUT_CH +
                 MNIST_CNN_FLAT +
                 MNIST_CNN_D1_UNITS +
                 10 + 10) * sizeof(float);

    size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%zu bytes (%.2f KB), buffers=%zu bytes (%.2f KB), total=%zu bytes (%.2f KB)\n",
           params_bytes, params_bytes / 1024.0,
           buffers_bytes, buffers_bytes / 1024.0,
           total_bytes, total_bytes / 1024.0);

    printf("sink=%f\n", sink);
    return 0;
}
