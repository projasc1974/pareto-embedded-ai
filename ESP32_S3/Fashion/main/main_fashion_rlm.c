#include <stdio.h>
#include <math.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "model_fashion_rlm.h"
#include "test_fashion_300.h"

static const char *TAG = "FASHION_RLM_NAIVE";

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30

// Softmax estable para 10 clases
static inline void softmax10(const float z[10], float p[10]) {
    float m = z[0];
    for (int i = 1; i < 10; i++) if (z[i] > m) m = z[i];
    float s = 0.0f;
    for (int i = 0; i < 10; i++) { p[i] = expf(z[i] - m); s += p[i]; }
    float inv = 1.0f / (s + 1e-12f);
    for (int i = 0; i < 10; i++) p[i] *= inv;
}

static inline int argmax10(const float p[10]) {
    int im = 0; float vm = p[0];
    for (int i = 1; i < 10; i++) if (p[i] > vm) { vm = p[i]; im = i; }
    return im;
}

// Inferencia RLM naive: x uint8[784] -> probs float[10]
static void rlm_infer_u8(const uint8_t x_u8[784], float probs_out[10]) {
    float z[10];

    for (int k = 0; k < 10; k++) {
        float acc = fashion_rlm_b[k];
        // dot(W[k], x_norm)
        for (int i = 0; i < 784; i++) {
            float x = ((float)x_u8[i]) / 255.0f;   // NAIVE normalize (igual que MNIST)
            acc += fashion_rlm_W[k][i] * x;
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

void app_main(void) {
    ESP_LOGI(TAG, "Fashion-MNIST RLM naive | test=%d | warmup=%d | measured=%d",
             FASHION_TEST300_N, WARMUP_PASSES, MEASURED_PASSES);

    // Heap baseline (nota: buffers estáticos/stack no aparecen aquí)
    size_t free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t min_before  = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

    // -------------------------
    // ACCURACY test300 (1 pasada)
    // -------------------------
    int correct = 0;
    for (int i = 0; i < FASHION_TEST300_N; i++) {
        float p10[10];
        rlm_infer_u8(fashion_test300_x[i], p10);
        int pred = argmax10(p10);
        if (pred == (int)fashion_test300_y[i]) correct++;
    }

    float p = (float)correct / (float)FASHION_TEST300_N;
    float acc = 100.0f * p;
    float acc_std = 0.0f; // determinista (test fijo)

    // IC 95% (aprox normal)
    float se = sqrtf((p * (1.0f - p)) / (float)FASHION_TEST300_N);
    float ci95 = 1.96f * se;
    float lo = p - ci95; if (lo < 0.0f) lo = 0.0f;
    float hi = p + ci95; if (hi > 1.0f) hi = 1.0f;

    // printf + log (por robustez)
    printf("ACCURACY test300: %.2f%% (std=%.2f) (%d/%d)\n", acc, acc_std, correct, FASHION_TEST300_N);
    printf("ACCURACY 95%% CI: [%.2f%%, %.2f%%]\n", 100.0f * lo, 100.0f * hi);

    ESP_LOGI(TAG, "ACCURACY test300: %.2f%% (std=%.2f) (%d/%d)", acc, acc_std, correct, FASHION_TEST300_N);
    ESP_LOGI(TAG, "ACCURACY 95%% CI: [%.2f%%, %.2f%%]", 100.0f * lo, 100.0f * hi);

    // -------------------------
    // Warm-up (descartado)
    // -------------------------
    float p10[10];
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < FASHION_TEST300_N; i++) {
            rlm_infer_u8(fashion_test300_x[i], p10);
        }
    }

    // -------------------------
    // Timing: MEASURED_PASSES
    // -------------------------
    double ms[MEASURED_PASSES];
    for (int k = 0; k < MEASURED_PASSES; k++) {
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < FASHION_TEST300_N; i++) {
            rlm_infer_u8(fashion_test300_x[i], p10);
        }
        int64_t t1 = esp_timer_get_time();
        ms[k] = (double)(t1 - t0) / 1000.0;
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)FASHION_TEST300_N;
    double thr_ips = 1000.0 / lat_ms_img;

    ESP_LOGI(TAG, "Batch mean: %.3f ms | std: %.3f ms", mean_ms, std_ms);
    ESP_LOGI(TAG, "Latency: %.6f ms/img | Throughput: %.2f img/s", lat_ms_img, thr_ips);

    // Peak heap used (aprox): (free_before - min_free_during)
    size_t min_after = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    size_t min_during = (min_after < min_before) ? min_after : min_before;
    size_t peak_used = (free_before > min_during) ? (free_before - min_during) : 0;

    ESP_LOGI(TAG, "Peak heap used (approx): %u bytes (%.2f KB)",
             (unsigned)peak_used, peak_used / 1024.0f);

    // -------------------------
    // Model memory estimate (params + buffers)
    // -------------------------
    // params: W(10x784) + b(10) float32
    size_t mem_params = sizeof(fashion_rlm_W) + sizeof(fashion_rlm_b);

    // buffers: logits z[10] + probs[10] (float32) + (p10[10] en stack, ya cuenta igual)
    // Para comparabilidad reportamos 20 floats:
    size_t mem_buffers = (10 * sizeof(float)) + (10 * sizeof(float));

    ESP_LOGI(TAG, "Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB)",
             (unsigned)mem_params, mem_params / 1024.0f,
             (unsigned)mem_buffers, mem_buffers / 1024.0f);

    (void)min_before; // silencio warnings si aplica
}
