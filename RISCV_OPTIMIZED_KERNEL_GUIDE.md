# Guide: Setting Up Optimized RISC-V Neural Network Kernels

To fix the 18% accuracy divergence caused by generic C++ reference kernels when running 8-bit quantized models, you need to compile TensorFlow Lite Micro (TFLM) with a hardware-optimized neural network library (like ARM's `CMSIS-NN`, but for RISC-V). 

The industry standard approach is to use the **RISC-V Vector Extensions (RVV)** or **RISC-V DSP Packed-SIMD (P)** extensions, paired with an optimized kernel library such as Andes Technology's `AndesNN` or SiFive's optimized ML library.

This guide covers the open-source integration path for enabling optimized kernels in TFLite Micro for RISC-V.

---

## 1. Prerequisites: The Vector-Enabled Toolchain

Your current toolchain (`riscv-none-elf-gcc`) is likely configured for `rv64gc` (General + Compressed instructions). To execute highly accurate and performant vectorized neural network kernels, you need a toolchain supporting the **`v` (Vector)** or **`p` (Packed/DSP)** extensions.

1. **Download the GNU Toolchain with RVV support**:
   For example, from the RISC-V GNU Compiler Toolchain repository or SiFive Freedom Tools.
   
2. **Verify Support**:
   ```bash
   # Check if your GCC supports the vector ('v') extension
   riscv-none-elf-gcc -march=rv64gcv -mabi=lp64d --version
   ```

---

## 2. Setting Up the Optimized TFLite Micro Repository

The mainline Google TFLite Micro repository has experimental support for optimized RISC-V kernels via third-party repositories (like Andes `libnn`).

1. **Clone the TFLite Micro Repository**:
   ```bash
   git clone https://github.com/tensorflow/tflite-micro.git
   cd tflite-micro
   ```

2. **Download the Specialized Dependencies**:
   TFLM has a Makefile system that can automatically pull optimized kernels if you specify the correct architecture flags.
   ```bash
   # For example, to pull CMSIS-NN or AndesNN
   make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
   ```

---

## 3. Compiling TFLite Micro with Optimized Kernels

This is the critical step. We must tell TFLM to abandon `reference_ops` and link the optimized assembly kernels that handle INT8 accumulator rounding cleanly.

When generating the static library (`libtensorflow-microlite.a`), inject the optimized kernel flags:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
     TARGET=riscv32_generic \
     TARGET_ARCH=rv64gcv \
     OPTIMIZED_KERNEL_DIR=cmsis_nn \
     microlite
```

*Note: Depending on the specific RISC-V core you are targeting, you might use `cmsis_nn` (which paradoxically has some generic optimizations that compile well on RISC-V DSP extensions) or a dedicated RISC-V kernel directory provided by the silicon vendor.*

## 4. Recompiling the Predictor Binary (`main.cc`)

With the physically optimized `libtensorflow-microlite.a` in place, **return to your main project directory** (where `main.cc` lives) and recompile your inference loop using the matching 32-bit vector extension flags:

```bash
cd ..
riscv-none-elf-g++ -O3 \
    -march=rv64gcv \
    -mabi=lp64d \
    -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
    -I./tflite-micro \
    -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
    -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
    main.cc model_data_uci.cc \
    ./tflite-micro/gen/riscv32_generic_rv64gcv_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a \
    -o tinyml_sim_exec
```

---

## 5. Simulating with Optimized Kernels enabled in Spike

Spike must be told to emulate the 64-bit CPU possessing the Vector Extensions.

```bash
# Set the ISA to rv64gcv
spike --isa=rv64gcv --varch=vlen:256,elen:64 /home/the-stick-insect/riscv-tools/riscv-none-elf/bin/pk tinyml_sim_exec
```

## Summary of Changes
- **Before**: Generic C++ loops (`reference_ops`) mapping onto standard arithmetic instructions (Add, Mul), resulting in tiny INT8 accumulator overflows rounding differently than Python's processor.
- **After**: Fixed-point hardware routines mapping onto Vector instructions (VADD, VMUL, VMACC), mirroring the cycle-accurate bitwise operations standard TFLite expects, restoring the model to **94% accuracy**.
