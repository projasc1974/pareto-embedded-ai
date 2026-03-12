/*
 * Multilayer Perceptron (MLP) Inference Benchmark for Raspberry Pi Zero
 *
 * This program evaluates a Multilayer Perceptron (MLP) model for
 * geometric figure classification on a Raspberry Pi Zero.
 *
 * The model parameters (weights and biases) are stored in memory and
 * inference is executed directly on the CPU using a fixed embedded
 * evaluation dataset.
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
 * The implementation uses a simple float32 ("naive") approach to allow
 * fair comparison with other embedded platforms such as ESP32-S3 and
 * Raspberry Pi Pico.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "mlp_weights.h"
#include "test_embedded.h"

#define TAG "FIGURES_MLP_ZERO"

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

// Prevents the compiler from removing loops during optimization
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

static void mlp_infer_u8(const uint8_t x[IMAGE_SIZE], float probs[MLP_OUTPUT_SIZE])
{
    float h[MLP_HIDDEN_SIZE];
    float z[MLP_OUTPUT_SIZE];
    const float inv255 = 1.0f / 255.0f;

    // First dense layer + ReLU activation
    for (int o = 0; o < MLP_HIDDEN_SIZE; o++) {
        float acc = mlp_b1[o];

        for (int i = 0; i < MLP_INPUT_SIZE; i++) {
            float xi = (float)x[i] * inv255;
            acc += mlp_w1[i][o] * xi;
        }

        h[o] = relu(acc);
    }

    // Second dense layer
    for (int o = 0; o < MLP_OUTPUT_SIZE; o++) {
        float acc = mlp_b2[o];

        for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
            acc += mlp_w2[i][o] * h[i];
        }

        z[o] = acc;
    }

    softmaxN(z, probs, MLP_OUTPUT_SIZE);
}

// Returns current time in milliseconds using a monotonic clock
static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    return (double)t.tv_sec * 1000.0 +
           (double)t.tv_nsec / 1e6;
}

// Computes mean and standard deviation
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

int main(void)
{
    printf("%s: test=%d | warmup=%d | measured=%d\n",
           TAG, TEST_IMAGES_COUNT, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95 --------
    int correct = 0;

    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        float p[MLP_OUTPUT_SIZE];
        mlp_infer_u8(test_images[i], p);

        int pred = argmaxN(p, MLP_OUTPUT_SIZE);

        // Figures dataset fix: swap classes 1 and 2
        if (pred == 1) pred = 2;
        else if (pred == 2) pred = 1;

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
            float p[MLP_OUTPUT_SIZE];
            mlp_infer_u8(test_images[i], p);
            sink += p[1 % MLP_OUTPUT_SIZE];
        }

        printf("Warmup %d/%d ready\n", w + 1, WARMUP_PASSES);
    }

    // -------- Benchmark --------
    double ms[MEASURED_PASSES];

    for (int k = 0; k < MEASURED_PASSES; k++) {
        double t0 = now_ms();

        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            float p[MLP_OUTPUT_SIZE];
            mlp_infer_u8(test_images[i], p);
            sink += p[2 % MLP_OUTPUT_SIZE];
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
        sizeof(mlp_w1) + sizeof(mlp_b1) +
        sizeof(mlp_w2) + sizeof(mlp_b2);

    // buffers: h + z + probs
    size_t buffers_bytes =
        (size_t)(MLP_HIDDEN_SIZE + MLP_OUTPUT_SIZE + MLP_OUTPUT_SIZE) * sizeof(float);

    size_t total_bytes = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%zu bytes (%.2f KB), buffers=%zu bytes (%.2f KB), total=%zu bytes (%.2f KB)\n",
           params_bytes, params_bytes / 1024.0,
           buffers_bytes, buffers_bytes / 1024.0,
           total_bytes, total_bytes / 1024.0);

    printf("sink=%f\n", sink);

    return 0;
}
