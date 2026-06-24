# Simulating TinyML Models on RISC-V using Spike

Since you have generated `<model>_int8.tflite` files and have downloaded Spike + RISC-V dependencies, here is the complete end-to-end workflow for running inference on your models through the Spike simulator.

## Prerequisites
Ensure the following are installed and on your system `$PATH`:
1. **RISC-V GNU Toolchain**: `riscv64-unknown-elf-gcc`, `riscv64-unknown-elf-g++`, etc.
2. **Spike Simulator**: `spike` (RISC-V ISA Simulator)
3. **RISC-V Proxy Kernel**: `pk` (needed by Spike to handle system calls like `printf`)
4. **xxd**: For converting your `.tflite` file into a C-byte array.
5. **TensorFlow Lite Micro (TFLM)**: You need the C++ source code for TFLM.

---

## Step 1: Convert the INT8 Model to a C Array
Microcontrollers and bare-metal RISC-V simulators don't have file systems (typically), so we embed the model directly into the C++ code.

```bash
cd /home/the-stick-insect/Human-Activity-Recognition-for-RISC-V
xxd -i models/har_realworld_cnn_tinyml_int8.tflite > model_data.cc
```
Open `model_data.cc` and add `const` to the array definition to place it in read-only memory, e.g.:
```cpp
const unsigned char models_har_realworld_cnn_tinyml_int8_tflite[] = { ... };
const unsigned int models_har_realworld_cnn_tinyml_int8_tflite_len = 25096;
```

---

## Step 2: Write the Inference Entry Point (`main.cc`)
You need a C++ file to initialize TensorFlow Lite Micro, load the model array, and run inference. Here's a minimized boilerplate specific to your model:

```cpp
#include <stdio.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Your model arrays from xxd
extern "C" {
  extern const unsigned char models_har_realworld_cnn_tinyml_int8_tflite[];
  void putchar_(char c) { putchar(c); }
}

// Arena size (tune this depending on your model's memory footprint)
const int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int main(int argc, char* argv[]) {
    tflite::InitializeTarget();
    printf("Starting RISC-V TinyML Inference Simulator...\n");

    // 1. Map the model
    const tflite::Model* model = tflite::GetModel(models_har_realworld_cnn_tinyml_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return 1;
    }

    // 2. Pull in only the necessary ops needed by your CNN + BatchNormalization
    tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D(); // Often used under the hood
    resolver.AddRelu();
    resolver.AddMaxPool2D(); // 1D is usually converted to 2D in TFLite
    resolver.AddAveragePool2D(); // For GlobalAvgPool
    resolver.AddFullyConnected(); // For Dense layers
    resolver.AddExpandDims();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMean();

    // 3. Build interpreter
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed.\n");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    // (Optional) Fill input->data.int8 with a sample vector from your dataset
    // For testing, let's just run it on 0s
    for (int i = 0; i < input->bytes; i++) {
        input->data.int8[i] = 0; // Quantized zero
    }

    // 4. Run Inference
    printf("Invoking model...\n");
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Invoke failed.\n");
        return 1;
    }
    
    // 5. Read output class
    int max_idx = 0;
    int8_t max_val = output->data.int8[0];
    // Assuming you have 8 classes (as seen in evaluate_tflite.py)
    for (int i = 1; i < 8; i++) {
        if (output->data.int8[i] > max_val) {
            max_val = output->data.int8[i];
            max_idx = i;
        }
    }

    printf("Predicted Class Index: %d\n", max_idx);
    printf("Simulation Completed Successfully!\n");
    return 0;
}
```

---

### 4. Compilation for RISC-V

Since your system-wide compiler is missing standard library headers, use the pre-built toolchain located in your home directory.

**Set Environment Variables:**
```bash
export RISCV_TOOLS=/home/the-stick-insect/riscv-tools
export PATH=$RISCV_TOOLS/bin:$PATH
export TFLM_ROOT=/home/the-stick-insect/Human-Activity-Recognition-for-RISC-V/tflite-micro
```

**Compile Command:**
```bash
riscv-none-elf-g++ -O3 -march=rv64gc -mabi=lp64d \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
  -I$TFLM_ROOT \
  -I$TFLM_ROOT/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I$TFLM_ROOT/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
  main.cc model_data.cc \
  $TFLM_ROOT/gen/riscv32_generic_rv64gc_default_gcc/lib/libtensorflow-microlite.a \
  -o tinyml_sim_exec
```

### 5. Running the Simulation

Use Spike with the Proxy Kernel (PK) to run the compiled binary:

```bash
spike /home/the-stick-insect/riscv-tools/riscv-none-elf/bin/pk tinyml_sim_exec
```

---

### Troubleshooting common issues

- **"stdio.h: No such file or directory"**: This happens if you use `/usr/bin/riscv64-unknown-elf-g++`. Always use the `riscv-none-elf-g++` version from your `~/riscv-tools/bin/` directory.
- **Symbol not found**: If you see missing symbols during linking, double check that all ops used in your model are added to the `MicroMutableOpResolver` in `main.cc`.
- **Model schema mismatch**: Ensure you are using the same version of TFLite Micro that was used to generate the `.tflite` model.
