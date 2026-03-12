#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "esp_err.h"

#include "mlp_weights.h"
#include "test_embedded.h"

#define TAG "FIGURES_MLP_NAIVE"

#define WARMUP_PASSES    5
#define MEASURED_PASSES  30
#define WDT_KICK_EVERY   1

static volatile float sink = 0.0f;

static inline void wdt_kick_every(int i, int every) {
    if ((i % every) == 0) {
        esp_task_wdt_reset();
    }
}

static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

static void softmaxN(const float *z, float *p, int n) {
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

static int argmaxN(const float *p, int n) {
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
  MLP naive:
    Input(784) -> Dense(64) -> ReLU -> Dense(3) -> Softmax

  IMPORTANTE:
  - Para Figuras + MLP sí aplica remapeo 1 <-> 2
  - Normalización correcta: x / 255.0
  - Layout esperado:
      mlp_w1[input][hidden]
      mlp_b1[hidden]
      mlp_w2[hidden][output]
      mlp_b2[output]
*/

static void mlp_infer_u8(const uint8_t x_u8[IMAGE_SIZE], float probs[MLP_OUTPUT_SIZE]) {
    float h[MLP_HIDDEN_SIZE];
    const float inv255 = 0.00392156886f;

    // Dense1 + ReLU
    for (int o = 0; o < MLP_HIDDEN_SIZE; o++) {
        float acc = mlp_b1[o];
        for (int i = 0; i < MLP_INPUT_SIZE; i++) {
            float x = (float)x_u8[i] * inv255;
            acc += mlp_w1[i][o] * x;
        }
        h[o] = relu(acc);
    }

    // Dense2
    float z[MLP_OUTPUT_SIZE];
    for (int o = 0; o < MLP_OUTPUT_SIZE; o++) {
        float acc = mlp_b2[o];
        for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
            acc += mlp_w2[i][o] * h[i];
        }
        z[o] = acc;
    }

    softmaxN(z, probs, MLP_OUTPUT_SIZE);
}

void app_main(void) {
    // WDT para benchmark largo
    esp_task_wdt_deinit();

    esp_task_wdt_config_t twdt_cfg = {
        .timeout_ms = 600000,   // 10 min
        .idle_core_mask = 0,    // no vigilar IDLE
        .trigger_panic = true
    };

    ESP_ERROR_CHECK(esp_task_wdt_init(&twdt_cfg));
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));

    ESP_LOGI(TAG, "FIGURES MLP naive | test=%d | warmup=%d | measured=%d",
             TEST_IMAGES_COUNT, WARMUP_PASSES, MEASURED_PASSES);

    size_t start_free = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t min_free_before = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

    // -------- Accuracy + CI95 --------
    int correct = 0;
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        wdt_kick_every(i, WDT_KICK_EVERY);

        float p[MLP_OUTPUT_SIZE];
        mlp_infer_u8(test_images[i], p);

        int pred = argmaxN(p, MLP_OUTPUT_SIZE);

        // FIX Figuras: remapeo 1 <-> 2
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

    ESP_LOGI(TAG, "ACCURACY test300: %.2f%% (std=0.00) (%d/%d)",
             100.0f * phat, correct, TEST_IMAGES_COUNT);
    ESP_LOGI(TAG, "ACCURACY 95%% CI: [%.2f%%, %.2f%%]",
             100.0f * lo, 100.0f * hi);

    // -------- Warm-up --------
    for (int w = 0; w < WARMUP_PASSES; w++) {
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            wdt_kick_every(i, WDT_KICK_EVERY);

            float p[MLP_OUTPUT_SIZE];
            mlp_infer_u8(test_images[i], p);
            sink += p[1 % MLP_OUTPUT_SIZE];
        }

        ESP_LOGI(TAG, "Warmup %d/%d listo", w + 1, WARMUP_PASSES);
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    // -------- Benchmark --------
    double ms[MEASURED_PASSES];

    for (int k = 0; k < MEASURED_PASSES; k++) {
        int64_t t0 = esp_timer_get_time();

        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            wdt_kick_every(i, WDT_KICK_EVERY);

            float p[MLP_OUTPUT_SIZE];
            mlp_infer_u8(test_images[i], p);
            sink += p[2 % MLP_OUTPUT_SIZE];
        }

        int64_t t1 = esp_timer_get_time();
        ms[k] = (double)(t1 - t0) / 1000.0;

        ESP_LOGI(TAG, "Pass %d/%d listo (%.1f ms)", k + 1, MEASURED_PASSES, ms[k]);
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    double mean_ms, std_ms;
    mean_std(ms, MEASURED_PASSES, &mean_ms, &std_ms);

    double lat_ms_img = mean_ms / (double)TEST_IMAGES_COUNT;
    double thr_ips = 1000.0 / lat_ms_img;

    ESP_LOGI(TAG, "Batch mean: %.3f ms | std: %.3f ms", mean_ms, std_ms);
    ESP_LOGI(TAG, "Latency: %.6f ms/img | Throughput: %.2f img/s",
             lat_ms_img, thr_ips);

    // -------- Peak heap --------
    size_t min_free_after = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    size_t peak_used = 0;
    if (min_free_after < start_free) {
        peak_used = start_free - min_free_after;
    }

    ESP_LOGI(TAG, "Peak heap used (approx): %u bytes (%.2f KB)",
             (unsigned)peak_used, (float)peak_used / 1024.0f);

    // -------- Memoria modelo --------
    const size_t params_bytes =
        sizeof(mlp_w1) + sizeof(mlp_b1) +
        sizeof(mlp_w2) + sizeof(mlp_b2);

    // buffers: h(64) + z(3) + probs(3)
    const size_t buffers_bytes =
        (size_t)(MLP_HIDDEN_SIZE + MLP_OUTPUT_SIZE + MLP_OUTPUT_SIZE) * sizeof(float);

    const size_t total_bytes = params_bytes + buffers_bytes;

    ESP_LOGI(TAG, "Model memory estimate: params=%u bytes (%.2f KB), buffers=%u bytes (%.2f KB), total=%u bytes (%.2f KB)",
             (unsigned)params_bytes,  (float)params_bytes / 1024.0f,
             (unsigned)buffers_bytes, (float)buffers_bytes / 1024.0f,
             (unsigned)total_bytes,   (float)total_bytes / 1024.0f);

    ESP_LOGI(TAG, "sink=%f | start_free=%u | min_free_before=%u | min_free_after=%u",
             (double)sink,
             (unsigned)start_free,
             (unsigned)min_free_before,
             (unsigned)min_free_after);

    ESP_LOGI(TAG, "Done.");

    ESP_ERROR_CHECK(esp_task_wdt_delete(NULL));
}
