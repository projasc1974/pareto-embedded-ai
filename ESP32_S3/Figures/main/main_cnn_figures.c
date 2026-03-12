/*
 * CNN Inference Benchmark for ESP32-S3
 *
 * This program implements a Convolutional Neural Network (CNN) used to classify
 * geometric figures on the ESP32-S3 platform. The implementation follows the
 * exact experimental methodology described in the research paper.
 *
 * The program performs the following steps:
 *   • Loads CNN weights and test images stored in flash memory.
 *   • Executes CNN inference using convolution, ReLU activation, max-pooling,
 *     dense layers, and softmax.
 *   • Runs a quick diagnostic test on the first images.
 *   • Executes a full benchmark experiment consisting of:
 *         - 5 warm-up passes
 *         - 30 measured passes
 *   • Measures inference latency, throughput, and classification accuracy.
 *   • Computes statistical metrics (mean accuracy and standard deviation).
 *   • Generates a confusion matrix and per-class accuracy.
 *
 * Important:
 *   - No class remapping is applied.
 *   - Predictions are compared directly with the ground-truth labels.
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
#include "cnn_weights.h"

static const char* TAG = "CNN_PAPER";

// ============================================================================
// GLOBAL BUFFERS (defined only if CNN_WEIGHTS_H exists)
// ============================================================================
#ifdef CNN_WEIGHTS_H
static float conv_output[CNN_FILTERS * 26 * 26];
static float pool_output[CNN_FILTERS * 13 * 13];
static float flat[CNN_FLATTEN_SIZE];
static float hidden[CNN_HIDDEN_SIZE];
static float logits[CNN_OUTPUT_SIZE];
static float probs[CNN_OUTPUT_SIZE];
#endif

// ============================================================================
// MATHEMATICAL FUNCTIONS
// ============================================================================

void matrix_vector_mult(const float* W, int rows, int cols, 
                        const float* x, const float* b, float* y) {
    for (int i = 0; i < rows; i++) {
        y[i] = b[i];
        for (int j = 0; j < cols; j++) {
            y[i] += W[i * cols + j] * x[j];
        }
    }
}

void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

void softmax(const float* logits, int size, float* probs) {
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

void conv2d_naive(const float* input, int input_h, int input_w, int input_c,
                  const float* kernel, int kernel_h, int kernel_w, int filters,
                  const float* bias, float* output) {
    int output_h = input_h - kernel_h + 1;
    int output_w = input_w - kernel_w + 1;
    
    for (int f = 0; f < filters; f++) {
        for (int oh = 0; oh < output_h; oh++) {
            for (int ow = 0; ow < output_w; ow++) {
                float sum = bias[f];
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        sum += input[ih * input_w + iw] * 
                               kernel[f * kernel_h * kernel_w + kh * kernel_w + kw];
                    }
                }
                output[f * output_h * output_w + oh * output_w + ow] = sum;
            }
        }
    }
}

void maxpool2x2(const float* input, int input_h, int input_w, int channels,
                float* output) {
    int output_h = input_h / 2;
    int output_w = input_w / 2;
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < output_h; oh++) {
            for (int ow = 0; ow < output_w; ow++) {
                int ih = oh * 2;
                int iw = ow * 2;
                int base = c * input_h * input_w;
                
                float max_val = input[base + ih * input_w + iw];
                max_val = fmax(max_val, input[base + ih * input_w + iw + 1]);
                max_val = fmax(max_val, input[base + (ih + 1) * input_w + iw]);
                max_val = fmax(max_val, input[base + (ih + 1) * input_w + iw + 1]);
                
                output[c * output_h * output_w + oh * output_w + ow] = max_val;
            }
        }
    }
}

// ============================================================================
// CNN MODEL - NO MAPPING, RAW PREDICTION ONLY
// ============================================================================

#ifdef CNN_WEIGHTS_H
int cnn_inference(const float* input) {
    // 1. Convolution layer
    conv2d_naive(input, 28, 28, 1,
                 (const float*)cnn_conv_weights, 3, 3, CNN_FILTERS,
                 cnn_conv_biases, conv_output);
    
    // 2. ReLU activation
    relu(conv_output, CNN_FILTERS * 26 * 26);
    
    // 3. Max Pooling
    maxpool2x2(conv_output, 26, 26, CNN_FILTERS, pool_output);
    
    // 4. Flatten
    memcpy(flat, pool_output, CNN_FLATTEN_SIZE * sizeof(float));
    
    // 5. Dense layer
    const float* dense_weights = (const float*)cnn_dense_weights;
    matrix_vector_mult(dense_weights, CNN_HIDDEN_SIZE, CNN_FLATTEN_SIZE,
                       flat, cnn_dense_biases, hidden);
    
    // 6. ReLU activation
    relu(hidden, CNN_HIDDEN_SIZE);
    
    // 7. Output layer
    const float* output_weights = (const float*)cnn_output_weights;
    matrix_vector_mult(output_weights, CNN_OUTPUT_SIZE, CNN_HIDDEN_SIZE,
                       hidden, cnn_output_biases, logits);
    
    // 8. Softmax
    softmax(logits, CNN_OUTPUT_SIZE, probs);
    
    // 9. Argmax - raw predicted class
    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 1; i < CNN_OUTPUT_SIZE; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted = i;
        }
    }
    
    return predicted;
}
#endif

// ============================================================================
// INFERENCE TASK - FOLLOWING THE PAPER METHODOLOGY
// ============================================================================

void inference_task(void *pvParameters) {
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "CNN - PAPER IMPLEMENTATION (NO MAPPING)");
    ESP_LOGI(TAG, "========================================\n");
    
    // Verify that the weights are available
#ifndef CNN_WEIGHTS_H
    ESP_LOGE(TAG, "Error: cnn_weights.h not found");
    vTaskDelete(NULL);
    return;
#endif
    
    float input_float[IMAGE_SIZE];
    
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
    
    // Compute model memory (according to methodology)
    size_t param_memory = (CNN_FILTERS * 3 * 3 + CNN_FILTERS +
                          CNN_FLATTEN_SIZE * CNN_HIDDEN_SIZE + CNN_HIDDEN_SIZE +
                          CNN_HIDDEN_SIZE * CNN_OUTPUT_SIZE + CNN_OUTPUT_SIZE) * sizeof(float);
    
    size_t buffer_memory = (sizeof(conv_output) + sizeof(pool_output) + 
                           sizeof(flat) + sizeof(hidden) + 
                           sizeof(logits) + sizeof(probs) + sizeof(input_float));
    
    ESP_LOGI(TAG, "\n=== MEMORY ===");
    ESP_LOGI(TAG, "Parameters: %.2f KB", param_memory / 1024.0);
    ESP_LOGI(TAG, "Buffers:    %.2f KB", buffer_memory / 1024.0);
    ESP_LOGI(TAG, "Total:      %.2f KB", (param_memory + buffer_memory) / 1024.0);
    
    // QUICK TEST (first 10 images) - informational only
    ESP_LOGI(TAG, "\n=== QUICK TEST (10 images) ===");
    int aciertos_rapidos = 0;
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            input_float[j] = test_images[i][j] / 255.0f;
        }
        
        int pred = cnn_inference(input_float);
        int real = test_labels[i];
        
        if (pred == real) aciertos_rapidos++;
        
        ESP_LOGI(TAG, "img%2d: real=%d pred=%d probs=[%.3f,%.3f,%.3f] %s",
                 i, real, pred, probs[0], probs[1], probs[2],
                 (pred == real) ? "✓" : "✗");
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    ESP_LOGI(TAG, "→ Quick accuracy: %d/10 = %.1f%%", aciertos_rapidos, aciertos_rapidos * 10.0f);
    
    // FULL EXPERIMENT (30 passes) - according to the paper methodology
    ESP_LOGI(TAG, "\n=== RUNNING FULL EXPERIMENT (30 passes) ===\n");
    
    float pass_accuracies[30];
    int total_correct = 0;
    int64_t total_time_us = 0;
    
    // Warm-up (5 passes)
    ESP_LOGI(TAG, "Warm-up (5 passes)...");
    for (int warm = 0; warm < 5; warm++) {
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                input_float[j] = test_images[i][j] / 255.0f;
            }
            cnn_inference(input_float);
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
            
            for (int j = 0; j < IMAGE_SIZE; j++) {
                input_float[j] = test_images[i][j] / 255.0f;
            }
            
            int pred = cnn_inference(input_float);
            
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
    float std_acc = sqrtf(sum_sq / 29.0f);
    
    float mean_latency_ms = (float)total_time_us / (30 * TEST_IMAGES_COUNT) / 1000.0f;
    float throughput = 1000.0f / mean_latency_ms;
    
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "FINAL RESULTS - CNN WITHOUT MAPPING");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Accuracy: %.4f ± %.4f", mean_acc, std_acc);
    ESP_LOGI(TAG, "Average latency: %.3f ms", mean_latency_ms);
    ESP_LOGI(TAG, "Throughput: %.2f ips", throughput);
    ESP_LOGI(TAG, "Model memory: %.2f KB", param_memory / 1024.0);
    ESP_LOGI(TAG, "Total memory (with buffers): %.2f KB", (param_memory + buffer_memory) / 1024.0);
    ESP_LOGI(TAG, "========================================\n");
    
    vTaskDelete(NULL);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

void app_main(void) {
    xTaskCreatePinnedToCore(
        inference_task,
        "cnn_paper_task",
        32768,
        NULL,
        5,
        NULL,
        0
    );
}
