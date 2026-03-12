/**
 * MLP Inference Benchmark for ESP32-S3
 *
 * This program evaluates a Multilayer Perceptron (MLP) model for geometric
 * figure classification on the ESP32-S3 platform.
 *
 * The model weights and test images are stored in flash memory. Images are
 * normalized on-the-fly using a temporary buffer in RAM in order to minimize
 * memory usage.
 *
 * The experiment follows the evaluation methodology used in the research paper:
 *
 *  • Warm-up phase to stabilize execution time
 *  • Multiple measured inference passes
 *  • Measurement of latency, throughput and accuracy
 *  • Statistical aggregation of results (mean and standard deviation)
 *
 * The dataset used for evaluation is stored in the header file `test_embedded.h`.
 */

#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "mlp_weights.h"
#include "test_embedded.h"

static const char *TAG = "MLP";

#define BATCH_SIZE TEST_IMAGES_COUNT  // 300 images
#define WARMUP_PASSES 5
#define MEASURED_PASSES 30
#define IMAGE_SIZE 784
#define NUM_CLASSES 3

// Temporary buffer for a single image (we do NOT store all normalized images)
float temp_image[IMAGE_SIZE];

// Activation functions
float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float *x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Forward propagation for a single image (input is float)
void forward(const float *input, float *output) {
    float hidden[MLP_HIDDEN_SIZE];
    
    // Hidden layer
    for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
        hidden[j] = mlp_b1[j];
        for (int i = 0; i < MLP_INPUT_SIZE; i++) {
            hidden[j] += input[i] * mlp_w1[i][j];
        }
        hidden[j] = relu(hidden[j]);
    }
    
    // Output layer
    for (int k = 0; k < MLP_OUTPUT_SIZE; k++) {
        output[k] = mlp_b2[k];
        for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
            output[k] += hidden[j] * mlp_w2[j][k];
        }
    }
    
    softmax(output, MLP_OUTPUT_SIZE);
}

// Evaluate the full batch - reads images directly from flash
float evaluate_batch(uint64_t *time_us) {
    float output[MLP_OUTPUT_SIZE];
    int correct = 0;
    
    uint64_t start = esp_timer_get_time();
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Normalize on-the-fly (one pixel at a time)
        for (int j = 0; j < IMAGE_SIZE; j++) {
            temp_image[j] = test_images[i][j] / 255.0f;
        }
        
        // Inference
        forward(temp_image, output);
        
        // Find predicted class
        int predicted = 0;
        for (int j = 1; j < MLP_OUTPUT_SIZE; j++) {
            if (output[j] > output[predicted]) predicted = j;
        }
        
        // Compare with ground-truth label
        if (predicted == test_labels[i]) {
            correct++;
        }
    }
    
    uint64_t end = esp_timer_get_time();
    *time_us = end - start;
    
    return (float)correct / BATCH_SIZE;
}

void mlp_experiment(void *pvParameters) {
    float latencies_ms[MEASURED_PASSES];
    float accuracies[MEASURED_PASSES];
    uint64_t batch_time_us;
    float accuracy;
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "MLP 784-64-3 - Experiment using flash data");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Configuration:");
    ESP_LOGI(TAG, "  - Batch size: %d images", BATCH_SIZE);
    ESP_LOGI(TAG, "  - Warm-up passes: %d", WARMUP_PASSES);
    ESP_LOGI(TAG, "  - Measured passes: %d", MEASURED_PASSES);
    ESP_LOGI(TAG, "========================================\n");
    
    // Verify class distribution
    int class_counts[NUM_CLASSES] = {0};
    for (int i = 0; i < BATCH_SIZE; i++) {
        class_counts[test_labels[i]]++;
    }
    ESP_LOGI(TAG, "Class distribution:");
    for (int i = 0; i < NUM_CLASSES; i++) {
        ESP_LOGI(TAG, "  Class %d: %d images", i, class_counts[i]);
    }
    
    // WARM-UP PASSES
    ESP_LOGI(TAG, "\nWarm-up (%d passes)...", WARMUP_PASSES);
    for (int pass = 0; pass < WARMUP_PASSES; pass++) {
        evaluate_batch(&batch_time_us);
        ESP_LOGI(TAG, "  Warm-up %d/%d completed", pass + 1, WARMUP_PASSES);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    // MEASURED PASSES
    ESP_LOGI(TAG, "\nMeasurements (%d passes)...", MEASURED_PASSES);
    
    float total_latency = 0;
    float total_accuracy = 0;
    
    for (int pass = 0; pass < MEASURED_PASSES; pass++) {
        accuracy = evaluate_batch(&batch_time_us);
        
        latencies_ms[pass] = batch_time_us / 1000.0f;
        accuracies[pass] = accuracy;
        
        total_latency += latencies_ms[pass];
        total_accuracy += accuracies[pass];
        
        ESP_LOGI(TAG, "  Pass %2d: total latency = %.3f ms (%.3f ms/img), accuracy = %.4f", 
                pass + 1, 
                latencies_ms[pass], 
                latencies_ms[pass] / BATCH_SIZE,
                accuracy);
        
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    // Compute statistics
    float mean_latency = total_latency / MEASURED_PASSES;
    float mean_accuracy = total_accuracy / MEASURED_PASSES;
    float latency_per_image = mean_latency / BATCH_SIZE;
    
    float std_latency = 0;
    float std_accuracy = 0;
    for (int pass = 0; pass < MEASURED_PASSES; pass++) {
        std_latency += powf(latencies_ms[pass] - mean_latency, 2);
        std_accuracy += powf(accuracies[pass] - mean_accuracy, 2);
    }
    std_latency = sqrtf(std_latency / MEASURED_PASSES);
    std_accuracy = sqrtf(std_accuracy / MEASURED_PASSES);
    
    float throughput = 1000.0f / latency_per_image;
    
    // Model memory (weights) + temporary buffer
    int weights_memory = (sizeof(mlp_w1) + sizeof(mlp_b1) + 
                          sizeof(mlp_w2) + sizeof(mlp_b2)) / 1024;
    int buffer_memory = sizeof(temp_image) / 1024;
    
    ESP_LOGI(TAG, "\n========================================");
    ESP_LOGI(TAG, "FINAL RESULTS - MLP on ESP32-S3");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Metric                Value");
    ESP_LOGI(TAG, "----------------------------------------");
    ESP_LOGI(TAG, "Accuracy (mean)       %.4f ± %.4f", mean_accuracy, std_accuracy);
    ESP_LOGI(TAG, "Total latency (mean)  %.3f ± %.3f ms", mean_latency, std_latency);
    ESP_LOGI(TAG, "Latency per image     %.3f ms", latency_per_image);
    ESP_LOGI(TAG, "Throughput            %.1f ips", throughput);
    ESP_LOGI(TAG, "Weights memory        %d KB", weights_memory);
    ESP_LOGI(TAG, "Temporary buffer      %d KB", buffer_memory);
    ESP_LOGI(TAG, "Estimated TOTAL memory %d KB", weights_memory + buffer_memory);
    ESP_LOGI(TAG, "========================================\n");
    
    // Table format for document
    ESP_LOGI(TAG, "Table format:");
    ESP_LOGI(TAG, "MLP-ESP32-S3 & %.4f & %.3f & %d & %.1f \\", 
             mean_accuracy, latency_per_image, weights_memory, throughput);
    
    vTaskDelete(NULL);
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting MLP experiment for ESP32-S3...");
    ESP_LOGI(TAG, "Using test_embedded.h with %d images", TEST_IMAGES_COUNT);
    ESP_LOGI(TAG, "Images stored in FLASH, only one temporary buffer in RAM");
    
    xTaskCreate(
        mlp_experiment,
        "mlp_exp",
        8192,
        NULL,
        5,
        NULL
    );
}
