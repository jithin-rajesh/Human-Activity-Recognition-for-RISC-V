#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "test_data.h"

extern "C" {
  extern const unsigned char models_har_cnn_tinyml_int8_tflite[];
  extern const unsigned int models_har_cnn_tinyml_int8_tflite_len;
  void putchar_(char c) { putchar(c); }
}

inline uint64_t rdinstret() {
    uint64_t val;
    asm volatile ("rdinstret %0" : "=r" (val));
    return val;
}

inline uint64_t rdcycle() {
    uint64_t val;
    asm volatile ("rdcycle %0" : "=r" (val));
    return val;
}

const int kTensorArenaSize = 1024 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Aligned model buffer to guarantee FlatBuffer metadata parsing
alignas(16) static uint8_t aligned_model_buf[32768]; // 32KB, larger than model

int main(int argc, char* argv[]) {
    tflite::InitializeTarget();
    printf("\n--- RISC-V TinyML Profiler (UCI HAR) ---\n");

    // Copy model into aligned buffer to fix FlatBuffer pointer parsing
    unsigned int model_len = models_har_cnn_tinyml_int8_tflite_len;
    printf("Model size: %u bytes, copying to aligned buffer at %p\n", model_len, (void*)aligned_model_buf);
    memcpy(aligned_model_buf, models_har_cnn_tinyml_int8_tflite, model_len);

    const tflite::Model* model = tflite::GetModel(aligned_model_buf);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return 1;
    }

    tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddMean();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed.\n");
        return 1;
    }
    printf("AllocateTensors OK. Arena used: %zu bytes\n", interpreter.arena_used_bytes());

    // Get raw data pointers - avoid reading any struct fields that may be misaligned
    int8_t* input_data = interpreter.input(0)->data.int8;
    int8_t* output_data = interpreter.output(0)->data.int8;
    printf("Input ptr: %p, Output ptr: %p\n", (void*)input_data, (void*)output_data);
    
    if (input_data == nullptr || output_data == nullptr) {
        printf("ERROR: Null data pointers!\n");
        return 1;
    }

    int correct = 0;
    uint64_t total_ins = 0;
    uint64_t total_cyc = 0;

    printf("Running %d samples...\n", kNumSamples);

    for (int s = 0; s < kNumSamples; s++) {
        // Write directly into the arena-managed input buffer
        for (int i = 0; i < kSampleDataSize; i++) {
            input_data[i] = batch_samples[s * kSampleDataSize + i];
        }

        uint64_t s_ins = rdinstret();
        uint64_t s_cyc = rdcycle();

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Invoke failed at %d\n", s);
            return 1;
        }

        uint64_t e_ins = rdinstret();
        uint64_t e_cyc = rdcycle();
        total_ins += (e_ins - s_ins);
        total_cyc += (e_cyc - s_cyc);

        // Re-read output pointer (Invoke may reallocate)
        output_data = interpreter.output(0)->data.int8;

        int pred = 0;
        int8_t best = output_data[0];
        for (int i = 1; i < 6; i++) {
            if (output_data[i] > best) {
                best = output_data[i];
                pred = i;
            }
        }

        if (s < 5) {
            printf("S%d: P=%d T=%d [", s, pred, batch_labels[s]);
            for (int i = 0; i < 6; i++) printf("%d ", output_data[i]);
            printf("]\n");
        }

        if (pred == batch_labels[s]) correct++;
    }

    float acc = (float)correct / kNumSamples * 100.0f;
    printf("\n--- Results ---\n");
    printf("Accuracy: %.2f%% (%d/%d)\n", acc, correct, kNumSamples);
    printf("RAM:      %zu bytes\n", interpreter.arena_used_bytes());
    printf("Avg Ins:  %llu\n", total_ins / kNumSamples);
    printf("Avg Cyc:  %llu\n", total_cyc / kNumSamples);
    printf("---------------\n");

    return 0;
}