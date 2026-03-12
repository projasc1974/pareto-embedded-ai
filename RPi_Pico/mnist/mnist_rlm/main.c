/*
 * Multinomial Logistic Regression (RLM) Inference Benchmark for Raspberry Pi Pico
 *
 * This program evaluates a multinomial logistic regression (RLM) model for
 * handwritten digit classification using the MNIST dataset on a Raspberry Pi Pico.
 *
 * The model parameters (weights and biases) are stored in flash memory and the
 * inference is executed directly on the microcontroller using a fixed evaluation
 * set of 300 images.
 *
 * The experiment follows the evaluation protocol used in the research paper:
 *
 *  • Accuracy evaluation on the entire test dataset
 *  • Computation of a 95% confidence interval assuming a binomial model
 *  • Warm-up phase to stabilize runtime behavior
 *  • 30 measured inference passes to estimate latency
 *  • Computation of throughput (images per second)
 *  • Estimation of model memory usage (parameters + buffers)
 *
 * The implementation intentionally uses a simple float32 ("naive") approach
 * without hardware acceleration in order to ensure fair comparison with
 * other embedded platforms such as ESP32-S3 and Raspberry Pi Zero.
 */

#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "model_mnist_rlm.h"
#include "test_mnist_300.h"

#define TAG "MNIST_RLM_NAIVE_PICO"

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

static volatile float sink = 0.0f;

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

static void rlm_infer(const float x[784], float probs[10]) {
    float z[10];

    for (int o = 0; o < OUTPUT_DIM; o++) {
        float acc = b[o];
        for (int i = 0; i < INPUT_DIM; i++) {
            acc += W[o][i] * x[i];
        }
        z[o] = acc;
    }

    softmax10(z, probs);
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

int main(void) {
    stdio_init_all();
    sleep_ms(3000);

    printf("%s: test=%d | warmup=%d | measured=%d\n",
           TAG, MNIST_TEST300_N, WARMUP_PASSES, MEASURED_PASSES);

    // -------- Accuracy + CI95% (single pass over the dataset) --------
    int correct = 0;

    for (int i = 0; i < MNIST_TEST300_N; i++) {
        float p10[10];
        rlm_infer(mnist_test300_x[i], p10);

        int pred = argmax10(p10);

        if (pred == (int)mnist_test300_y[i]) correct++;

        sink += p10[0];
    }

    float p = (float)correct / (float)MNIST_TEST300_N;

    float se   = sqrtf((p * (1.0f - p)) / (float)MNIST_TEST300_N);
    float ci95 = 1.96f * se;

    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    printf("ACCURACY test300: %.2f%% (std=0.00) (%d/%d)\n",
           100.0f * p, correct, MNIST_TEST300_N);

    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n",
           100.0f * lo, 100.0f * hi);

    // -------- Warm-up phase --------
    for (int w = 0; w < WARMUP_PASSES; w++) {

        for (int i = 0; i < MNIST_TEST300_N; i++) {

            float p10[10];

            rlm_infer(mnist_test300_x[i], p10);

            sink += p10[1];
        }
    }

    // -------- Timing measurements --------
    double ms[MEASURED_PASSES];

    for (int k = 0; k < MEASURED_PASSES; k++) {

        uint64_t t0 = time_us_64();

        for (int i = 0; i < MNIST_TEST300_N; i++) {

            float p10[10];

            rlm_infer(mnist_test300_x[i], p10);

            sink += p10[2];
        }

        uint64_t t1 = time_us_64();

        ms[k] = (double)(t1 - t0) / 1000.0;
    }

    double mean_ms, std_ms;

    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)MNIST_TEST300_N;
    double thr_ips = 1000.0 / lat_ms_img;

    printf("Batch mean: %.3f ms | std: %.3f ms\n", mean_ms, std_ms);
    printf("Latency: %.6f ms/img | Throughput: %.2f img/s\n", lat_ms_img, thr_ips);

    // -------- Model memory usage (parameters + buffers + total) --------
    const size_t params_bytes  = sizeof(W) + sizeof(b);
    const size_t buffers_bytes = (10 * sizeof(float)) + (10 * sizeof(float)); // z + probs
    const size_t total_bytes   = params_bytes + buffers_bytes;

    printf("Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)\n",
           (unsigned)params_bytes,  params_bytes / 1024.0f,
           (unsigned)buffers_bytes, buffers_bytes / 1024.0f,
           (unsigned)total_bytes,   total_bytes / 1024.0f);

    printf("sink=%f\n", (double)sink);

    while (true) { tight_loop_contents(); }

    return 0;
}
