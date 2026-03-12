/*
 * main_cnn_only.c - Implementación CNN para ESP32-S3
 * Siguiendo estrictamente la metodología del paper
 * SIN mapeo inteligente - solo predicción raw
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
// BUFFERS GLOBALES (definidos solo si CNN_WEIGHTS_H existe)
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
// FUNCIONES MATEMÁTICAS
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
// MODELO CNN - SIN MAPEO, SOLO PREDICCIÓN RAW
// ============================================================================

#ifdef CNN_WEIGHTS_H
int cnn_inference(const float* input) {
    // 1. Capa convolucional
    conv2d_naive(input, 28, 28, 1,
                 (const float*)cnn_conv_weights, 3, 3, CNN_FILTERS,
                 cnn_conv_biases, conv_output);
    
    // 2. ReLU
    relu(conv_output, CNN_FILTERS * 26 * 26);
    
    // 3. Max Pooling
    maxpool2x2(conv_output, 26, 26, CNN_FILTERS, pool_output);
    
    // 4. Flatten
    memcpy(flat, pool_output, CNN_FLATTEN_SIZE * sizeof(float));
    
    // 5. Capa densa
    const float* dense_weights = (const float*)cnn_dense_weights;
    matrix_vector_mult(dense_weights, CNN_HIDDEN_SIZE, CNN_FLATTEN_SIZE,
                       flat, cnn_dense_biases, hidden);
    
    // 6. ReLU
    relu(hidden, CNN_HIDDEN_SIZE);
    
    // 7. Capa de salida
    const float* output_weights = (const float*)cnn_output_weights;
    matrix_vector_mult(output_weights, CNN_OUTPUT_SIZE, CNN_HIDDEN_SIZE,
                       hidden, cnn_output_biases, logits);
    
    // 8. Softmax
    softmax(logits, CNN_OUTPUT_SIZE, probs);
    
    // 9. Argmax - clase predicha RAW
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
// TAREA DE INFERENCIA - SIGUIENDO LA METODOLOGÍA DEL PAPER
// ============================================================================

void inference_task(void *pvParameters) {
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "CNN - IMPLEMENTACIÓN PAPER (SIN MAPEO)");
    ESP_LOGI(TAG, "========================================\n");
    
    // Verificar que tenemos los pesos
    #ifndef CNN_WEIGHTS_H
    ESP_LOGE(TAG, "Error: cnn_weights.h no encontrado");
    vTaskDelete(NULL);
    return;
    #endif
    
    float input_float[IMAGE_SIZE];
    
    // Verificar distribución de etiquetas
    int clase_counts[3] = {0, 0, 0};
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        clase_counts[test_labels[i]]++;
    }
    
    ESP_LOGI(TAG, "\n=== VERIFICACIÓN DE ETIQUETAS ===");
    ESP_LOGI(TAG, "Clase 0: %d imágenes", clase_counts[0]);
    ESP_LOGI(TAG, "Clase 1: %d imágenes", clase_counts[1]);
    ESP_LOGI(TAG, "Clase 2: %d imágenes", clase_counts[2]);
    ESP_LOGI(TAG, "Total: %d imágenes", TEST_IMAGES_COUNT);
    
    // Calcular memoria del modelo (según metodología)
    size_t param_memory = (CNN_FILTERS * 3 * 3 + CNN_FILTERS +
                          CNN_FLATTEN_SIZE * CNN_HIDDEN_SIZE + CNN_HIDDEN_SIZE +
                          CNN_HIDDEN_SIZE * CNN_OUTPUT_SIZE + CNN_OUTPUT_SIZE) * sizeof(float);
    
    size_t buffer_memory = (sizeof(conv_output) + sizeof(pool_output) + 
                           sizeof(flat) + sizeof(hidden) + 
                           sizeof(logits) + sizeof(probs) + sizeof(input_float));
    
    ESP_LOGI(TAG, "\n=== MEMORIA ===");
    ESP_LOGI(TAG, "Parámetros: %.2f KB", param_memory / 1024.0);
    ESP_LOGI(TAG, "Buffers:    %.2f KB", buffer_memory / 1024.0);
    ESP_LOGI(TAG, "Total:      %.2f KB", (param_memory + buffer_memory) / 1024.0);
    
    // PRUEBA RÁPIDA (primeras 10 imágenes) - solo informativa
    ESP_LOGI(TAG, "\n=== PRUEBA RÁPIDA (10 imágenes) ===");
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
    
    ESP_LOGI(TAG, "→ Accuracy rápida: %d/10 = %.1f%%", aciertos_rapidos, aciertos_rapidos * 10.0f);
    
    // EXPERIMENTO COMPLETO (30 pasadas) - según metodología del paper
    ESP_LOGI(TAG, "\n=== EJECUTANDO EXPERIMENTO COMPLETO (30 pasadas) ===\n");
    
    float pass_accuracies[30];
    int total_correct = 0;
    int64_t total_time_us = 0;
    
    // Warm-up (5 pasadas) - según metodología
    ESP_LOGI(TAG, "Calentamiento (5 pasadas)...");
    for (int warm = 0; warm < 5; warm++) {
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                input_float[j] = test_images[i][j] / 255.0f;
            }
            cnn_inference(input_float);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    // 30 pasadas medidas
    for (int pass = 0; pass < 30; pass++) {
        int pass_correct = 0;
        int64_t pass_start = esp_timer_get_time();
        
        for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
            // Pequeña pausa cada 50 imágenes para evitar WDT
            if (i % 50 == 0 && i > 0) {
                vTaskDelay(pdMS_TO_TICKS(1));
            }
            
            for (int j = 0; j < IMAGE_SIZE; j++) {
                input_float[j] = test_images[i][j] / 255.0f;
            }
            
            int pred = cnn_inference(input_float);
            
            // SIN MAPEO - comparación directa con etiqueta real
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
    
    // Calcular métricas finales
    float sum_acc = 0;
    for (int i = 0; i < 30; i++) sum_acc += pass_accuracies[i];
    float mean_acc = sum_acc / 30.0f;
    
    float sum_sq = 0;
    for (int i = 0; i < 30; i++) {
        float diff = pass_accuracies[i] - mean_acc;
        sum_sq += diff * diff;
    }
    float std_acc = sqrtf(sum_sq / 29.0f);  // desviación estándar muestral
    
    float mean_latency_ms = (float)total_time_us / (30 * TEST_IMAGES_COUNT) / 1000.0f;
    float throughput = 1000.0f / mean_latency_ms;
    
    // RESULTADOS FINALES - formato similar al paper
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "RESULTADOS FINALES - CNN SIN MAPEO");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Accuracy: %.4f ± %.4f", mean_acc, std_acc);
    ESP_LOGI(TAG, "Latencia media: %.3f ms", mean_latency_ms);
    ESP_LOGI(TAG, "Throughput: %.2f ips", throughput);
    ESP_LOGI(TAG, "Memoria del modelo: %.2f KB", param_memory / 1024.0);
    ESP_LOGI(TAG, "Memoria total (con buffers): %.2f KB", (param_memory + buffer_memory) / 1024.0);
    ESP_LOGI(TAG, "========================================\n");
    
    // Matriz de confusión
    ESP_LOGI(TAG, "\n=== MATRIZ DE CONFUSIÓN ===");
    int confusion[3][3] = {{0}};
    for (int i = 0; i < TEST_IMAGES_COUNT; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            input_float[j] = test_images[i][j] / 255.0f;
        }
        int pred = cnn_inference(input_float);
        confusion[test_labels[i]][pred]++;
    }
    
    ESP_LOGI(TAG, "      Pred 0  Pred 1  Pred 2");
    for (int r = 0; r < 3; r++) {
        ESP_LOGI(TAG, "Real %d:  %3d     %3d     %3d", 
                 r, confusion[r][0], confusion[r][1], confusion[r][2]);
    }
    
    // Accuracy por clase
    for (int i = 0; i < 3; i++) {
        int total = clase_counts[i];
        int correct = confusion[i][i];
        float class_acc = (total > 0) ? (float)correct / total : 0;
        ESP_LOGI(TAG, "Accuracy clase %d: %.2f%% (%d/%d)", 
                 i, class_acc * 100, correct, total);
    }
    
    // Formato para tabla (como en el paper)
    ESP_LOGI(TAG, "\n📋 Resultado para tabla (formato paper):");
    ESP_LOGI(TAG, "| CNN-ESP32-S3-Figures | %.4f ± %.4f | %.3f | %.2f | %.2f |",
             mean_acc, std_acc, mean_latency_ms, param_memory / 1024.0, throughput);
    
    vTaskDelete(NULL);
}

// ============================================================================
// FUNCIÓN PRINCIPAL
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
