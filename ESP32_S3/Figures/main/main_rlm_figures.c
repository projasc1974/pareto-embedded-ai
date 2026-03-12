/*
 * MLR Inference Benchmark for ESP32-S3
 *
 * This program implements a Multinomial Logistic Regression (MLR) model
 * for geometric figure classification on the ESP32-S3 platform.
 * The implementation follows the experimental methodology defined in the paper.
 *
 * Main tasks performed by this program:
 * - Load model weights and embedded test images from header files.
 * - Run inference using matrix-vector multiplication and softmax.
 * - Perform a quick diagnostic test on the first 10 images.
 * - Execute the full benchmark experiment with:
 *      - 5 warm-up passes
 *      - 30 measured passes
 * - Compute final metrics including:
 *      - mean accuracy
 *      - standard deviation
 *      - average latency
 *      - throughput
 *      - memory usage
 * - Generate a confusion matrix and per-class accuracy.
 *
 * Important:
 * - No class remapping is applied.
 * - Raw predictions are compared directly against the ground-truth labels.
 */

/*
 * main_mlr_figuras.c - MLR implementation for ESP32-S3
 * Strictly following the methodology described in the paper
 * WITHOUT intelligent mapping - raw prediction only
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"

// Headers
#include "test_embedded.h"
#include "rlm_weights.h"  // MLR = Multinomial Logistic Regression

static const char* TAG = "MLR_PAPER";

// ============================================================================
// CONSTANTS (taken from the weights file)
// ============================================================================
#ifndef RLM_CLASSES
#define RLM_CLASSES 3
#endif

#ifndef RLM_FEATURES
#define RLM_FEATURES 784
#endif

// ============================================================================
// GLOBAL BUFFERS
// ============================================================================
static float mlr_logits[RLM_CLASSES];
static float mlr_probs[RLM_CLASSES];

// ============================================================================
// MATHEMATICAL FUNCTIONS (identical to those used in the CNN implementation)
// ============================================================================

void matrix_vector_mult_mlr(const float* W, int rows, int cols, 
                            const float* x, const float* b, float* y) {
    for (int i = 0; i < rows; i++) {
        y[i] = b[i];
        for (int j = 0; j < cols; j++) {
            // W is organized as [rows][cols] in the header file
            y[i] += W[i * cols + j] * x[j];
        }
    }
}

void softmax_mlr(const float* logits, int size, float* probs) {
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    
    for (int i = 0; i < size; i++) {
        probs[i] /= sum;
    }
}

// ============================================================================
// MLR MODEL (Multinomial Logistic Regression)
// ============================================================================

int mlr_inference(const float* input) {
    // Matrix-vector product: W [3×784] × x [784] + bias
    // The weights are stored in rlm_weights[3][784] (from the header file)
    // IMPORTANT: rlm_weights is organized as [classes][features]
    matrix_vector_mult_mlr((const float*)rlm_weights, 
                          RLM_CLASSES, RLM_FEATURES,
                          input, rlm_biases, mlr_logits);
    
    // Softmax to obtain probabilities
    softmax_mlr(mlr_logits, RLM_CLASSES, mlr_probs);
    
    // Argmax to obtain the predicted class
    int predicted = 0;
    float max_prob = mlr_probs[0];
    for (int i = 1; i < RLM_CLASSES; i++) {
        if (mlr_probs[i] > max_prob) {
            max_prob = mlr_probs[i];
            predicted = i;
        }
    }
    
    return predicted;
}

// ============================================================================
// INFERENCE TASK - FOLLOWING THE PAPER METHODOLOGY
// ============================================================================

void inference_task(void *pvParameters) {
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "MLR - PAPER IMPLEMENTATION (NO MAPPING)");
    ESP_LOGI(TAG, "========================================\n");
    
    float input_float[RLM_FEATURES];
    
    // Verify label distribution
    int clase_counts[3] = {0, 0, 0};
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        clase_counts[test_labels[i]]++;
    }
    
    ESP_LOGI(TAG, "\n=== LABEL VERIFICATION ===");
    ESP_LOGI(TAG, "Class 0: %d images", clase_counts[0]);
    ESP_LOGI(TAG, "Class 1: %d images", clase_counts[1]);
    ESP_LOGI(TAG, "Class 2: %d images", clase_counts[2]);
    ESP_LOGI(TAG, "Total: %d images", TEST_IMAGES_COUNT);
    
    // Compute model memory usage (according to methodology)
    size_t param_memory = (RLM_CLASSES * RLM_FEATURES + RLM_CLASSES) * sizeof(float);
    size_t buffer_memory = sizeof(input_float) + sizeof(mlr_logits) + sizeof(mlr_probs);
    
    ESP_LOGI(TAG, "\n=== MEMORY ===");
    ESP_LOGI(TAG, "Parameters: %.2f KB", param_memory / 1024.0);
    ESP_LOGI(TAG, "Buffers:    %.2f KB", buffer_memory / 1024.0);
    ESP_LOGI(TAG, "Total:      %.2f KB", (param_memory + buffer_memory) / 1024.0);
    ESP_LOGI(TAG, "Detail: W[%d][%d] + b[%d]", RLM_CLASSES, RLM_FEATURES, RLM_CLASSES);
    
    // QUICK TEST (first 10 images) - informational only
    ESP_LOGI(TAG, "\n=== QUICK TEST (10 images) ===");
    int aciertos_rapidos = 0;
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < RLM_FEATURES; j++) {
            input_float[j] = test_images[i][j] / 255.0f;
        }
        
        int pred = mlr_inference(input_float);
        int real = test_labels[i];
        
        if (pred == real) aciertos_rapidos++;
        
        ESP_LOGI(TAG, "img%2d: real=%d pred=%d probs=[%.3f,%.3f,%.3f] %s",
                 i, real, pred, mlr_probs[0], mlr_probs[1], mlr_probs[2],
                 (pred == real) ? "✓" : "✗");
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    ESP_LOGI(TAG, "→ Quick accuracy: %d/10 = %.1f%%", aciertos_rapidos, aciertos_rapidos * 10.0f);
    
    // FULL EXPERIMENT (30 passes) - according to the paper methodology
    ESP_LOGI(TAG, "\n=== RUNNING FULL EXPERIMENT (30 passes) ===\n");
    
    float pass_accuracies[30];
    int total_correct = 0;
    int64_t total_time_us = 0;
    
    // Warm-up (5 passes) - according to methodology
    ESP_LOGI(TAG, "Warm-up (5 passes)...");
    for (int warm = 0; warm < 5; warm++) {
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            for (int j = 0; j < RLM_FEATURES; j++) {
                input_float[j] = test_images[i][j] / 255.0f;
            }
            mlr_inference(input_float);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    // 30 measured passes
    for (int pass = 0; pass < 30; pass++) {
        int pass_correct = 0;
        int64_t pass_start = esp_timer_get_time();
        
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            // Small pause every 50 images to avoid WDT
            if (i % 50 == 0 && i > 0) {
                vTaskDelay(pdMS_TO_TICKS(1));
            }
            
            for (int j = 0; j < RLM_FEATURES; j++) {
                input_float[j] = test_images[i][j] / 255.0f;
            }
            
            int pred = mlr_inference(input_float);
            
            // NO MAPPING - direct comparison with ground-truth label
            if (pred == test_labels[i]) {
                pass_correct++;
            }
        }
        
        int64_t pass_end = esp_timer_get_time();
        int64_t pass_time = pass_end - pass_start;
        
        float pass_acc = (float)pass_correct / TEST_IMAGES_COUNT;
        pass_accuracies[pass] = pass_acc;
        total_time_us += pass_time;
        total_correct += pass_correct;
        
        ESP_LOGI(TAG, "  Pass %2d: acc=%.4f (%d/%d) time=%lld ms",
                 pass+1, pass_acc, pass_correct, TEST_IMAGES_COUNT, pass_time/1000);
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    // Compute final metrics
    float sum_acc = 0;
    for (int i = 0; i < 30; i++) sum_acc += pass_accuracies[i];
    float mean_acc = sum_acc / 30.0f;
    
    float sum_sq = 0;
    for (int i = 0; i < 30; i++) {
        float diff = pass_accuracies[i] - mean_acc;
        sum_sq += diff * diff;
    }
    float std_acc = sqrtf(sum_sq / 29.0f);  // sample standard deviation
    
    float mean_latency_ms = (float)total_time_us / (30 * TEST_IMAGES_COUNT) / 1000.0f;
    float throughput = 1000.0f / mean_latency_ms;
    
    // FINAL RESULTS - format similar to the paper
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "FINAL RESULTS - MLR WITHOUT MAPPING");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Accuracy: %.4f ± %.4f", mean_acc, std_acc);
    ESP_LOGI(TAG, "Average latency: %.3f ms", mean_latency_ms);
    ESP_LOGI(TAG, "Throughput: %.2f ips", throughput);
    ESP_LOGI(TAG, "Model memory: %.2f KB", param_memory / 1024.0);
    ESP_LOGI(TAG, "Total memory (with buffers): %.2f KB", (param_memory + buffer_memory) / 1024.0);
    ESP_LOGI(TAG, "========================================\n");
    
    // Confusion matrix
    ESP_LOGI(TAG, "\n=== CONFUSION MATRIX ===");
    int confusion[3][3] = {{0}};
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        for (int j = 0; j < RLM_FEATURES; j++) {
            input_float[j] = test_images[i][j] / 255.0f;
        }
        int pred = mlr_inference(input_float);
        confusion[test_labels[i]][pred]++;
    }
    
    ESP_LOGI(TAG, "      Pred 0  Pred 1  Pred 2");
    for (int r = 0; r < 3; r++) {
        ESP_LOGI(TAG, "Real %d:  %3d     %3d     %3d", 
                 r, confusion[r][0], confusion[r][1], confusion[r][2]);
    }
    
    // Per-class accuracy
    for (int i = 0; i < 3; i++) {
        int total = clase_counts[i];
        int correct = confusion[i][i];
        float class_acc = (total > 0) ? (float)correct / total : 0;
        ESP_LOGI(TAG, "Class %d accuracy: %.2f%% (%d/%d)", 
                 i, class_acc * 100, correct, total);
    }
    
    // Table-ready format (as in the paper)
    ESP_LOGI(TAG, "\n Table-ready result (paper format):");
    ESP_LOGI(TAG, "| MLR-ESP32-S3-Figures | %.4f ± %.4f | %.3f | %.2f | %.2f |",
             mean_acc, std_acc, mean_latency_ms, param_memory / 1024.0, throughput);
    
    vTaskDelete(NULL);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

void app_main(void) {
    xTaskCreatePinnedToCore(
        inference_task,
        "mlr_paper_task",
        32768,
        NULL,
        5,
        NULL,
        0
    );
}
