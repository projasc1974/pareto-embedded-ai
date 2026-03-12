/*
 * Multilayer Perceptron (MLP) Inference Benchmark for Raspberry Pi Zero
 *
 * This program evaluates a Multilayer Perceptron (MLP) model for
 * Fashion-MNIST image classification using a Raspberry Pi Zero.
 *
 * The neural network parameters (weights and biases) are stored in memory
 * and inference is executed directly on the CPU using a fixed embedded
 * dataset consisting of 300 Fashion-MNIST images.
 *
 * The experimental protocol follows the methodology used in the study:
 *
 *  • Accuracy evaluation on the complete test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize execution behavior
 *  • 30 measured inference passes for latency estimation
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * The implementation uses a naive float32 computation approach
 * to ensure fair comparison with other embedded platforms.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "model_fashion_mlp.h"
#include "test_fashion_300.h"

#define TAG "FASHION_MLP_ZERO"
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

static void mlp_infer_u8(const uint8_t x[784], float probs[10]) {
    float h[FASHION_MLP_HIDDEN];
    float z[10];
    const float inv255 = 1.0f / 255.0f;

    // First dense layer followed by ReLU activation
    for (int o = 0; o < FASHION_MLP_HIDDEN; o++) {
        float acc = fashion_mlp_b1[o];
        const float *w = &fashion_mlp_w1[o * 784];

        for (int i = 0; i < 784; i++) {
            acc += w[i] * ((float)x[i] * inv255);
        }

        h[o] = relu(acc);
    }

    // Output dense layer
    for (int o = 0; o < 10; o++) {
        float acc = fashion_mlp_b2[o];
        const float *w = &fashion_mlp_w2[o * FASHION_MLP_HIDDEN];

        for (int i = 0; i < FASHION_MLP_HIDDEN; i++) {
            acc += w[i] * h[i];
        }

        z[o] = acc;
    }

    softmax10(z, probs);
}

// Returns current time in milliseconds using a monotonic clock
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    return (double)t.tv_sec * 1000.0 +
           (double)t.tv_nsec / 1e6;
}

// Computes mean and standard deviation of measured times
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

int main(void) {
    printf("%s: test=%d | warmup=%d | measured=%d | hidden=%d\n",
           TAG, FASHION_TEST300_N, WARMUP_PASSES, MEASURED_PASSES, FASHION_MLP_HIDDEN);

    // -------- Accuracy + CI95 --------
    int correct = 0;

    for (int i = 0; i < FASHION_TEST300_N; i++) {
        float p[10];
        mlp_infer_u8(fashion_test300_x[i], p);

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
            mlp_infer_u8(fashion_test300_x[i], p);
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
            mlp_infer_u8(fashion_test300_x[i], p);
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
        sizeof(fashion_mlp_w1) + sizeof(fashion_mlp_b1) +
        sizeof(fashion_mlp_w2) + sizeof(fashion_mlp_b2);

    // Hidden activations + logits + probabilities
    size_t buffers_bytes =
        (size_t)(FASHION_MLP_HIDDEN + 10 + 10) * sizeof(float);

    size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%zu bytes (%.2f KB), buffers=%zu bytes (%.2f KB), total=%zu bytes (%.2f KB)\n",
           params_bytes, params_bytes / 1024.0,
           buffers_bytes, buffers_bytes / 1024.0,
           total_bytes, total_bytes / 1024.0);

    printf("sink=%f\n", sink);

    return 0;
}
