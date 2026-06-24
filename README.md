# Human Activity Recognition for RISC-V

This repository contains the complete pipeline for training, quantizing, and deploying Human Activity Recognition (HAR) models on RISC-V architectures. The project utilizes deep learning architectures (CNN, GRU, LSTM) and TensorFlow Lite for Microcontrollers (TFLite Micro) to enable efficient, edge-based inference.

## Overview

The goal of this project is to create highly optimized HAR models that can run on resource-constrained RISC-V microcontrollers. The pipeline covers:
1. **Data Preprocessing**: Handling and scaling raw sensor data (e.g., accelerometer, gyroscope).
2. **Model Training & Evaluation**: Training models like CNN and CNN-GRU, including Leave-One-Subject-Out (LOSO) cross-validation for robustness.
3. **Quantization**: Converting models to `INT8` format via TFLite to dramatically reduce memory footprint and latency.
4. **RISC-V Simulation & Execution**: Compiling the quantized models with TFLite Micro and running them in a RISC-V environment (using simulators like Spike).
5. **Real-time Demonstrations**: Providing websocket-based streaming tools to visualize activity recognition in real-time.

## Documentation & Guides

For detailed instructions on specific parts of the project, please refer to the following guides:

- **[Dataset Guide](DATASET_GUIDE.md)**: Details on the datasets used, preprocessing steps, and data structures.
- **[Spike Simulation Guide](SPIKE_SIMULATION_GUIDE.md)**: Instructions on how to set up and run the RISC-V Spike simulator for TFLite Micro model execution.
- **[RISC-V Optimized Kernel Guide](RISCV_OPTIMIZED_KERNEL_GUIDE.md)**: Information regarding the underlying optimized neural network kernels for RISC-V.
- **[Layman Simulation Guide](LAYMAN_SIMULATION_GUIDE.md)**: A high-level overview of running the simulation for users without deep technical backgrounds.
- **[Model Analysis Report](model_analysis_report.md)**: Detailed metrics, confusion matrices, and analysis of model performances.

## Repository Structure

- `train_*.py` / `evaluate_*.py`: Scripts used to train the Keras models, evaluate their accuracy, and convert/quantize them to TFLite formats.
- `main.cc`, `model_data.cc`, `test_data.h`: C++ source files for initializing TFLite Micro, loading the quantized model arrays, and performing inference.
- `models/`: Contains the generated `.keras` and `.tflite` model files along with their training history and performance graphs.
- `websocket_har_demo/`: A real-time web application to stream sensor data (e.g., via Phyphox) and visualize live predictions.
- `exhibition_showcase/`: Front-end components for demonstrating the project's capabilities.

## Getting Started

### 1. Training Models
To train the CNN model with quantization-aware strategies or standard pipelines, use the provided Python scripts:
```bash
python train_cnn_tinyml.py
python train_realworld_cnn_loso.py
```
Models will be evaluated and exported as `.keras` and `.tflite` artifacts in the `models/` directory.

### 2. C++ Simulation
Export the quantized TFLite model to a C-byte array (e.g., `model_data.cc`). Then, compile the `main.cc` with the TFLite Micro library and run the executable in your RISC-V simulator.
Refer to the [Spike Simulation Guide](SPIKE_SIMULATION_GUIDE.md) for detailed compilation commands.

### 3. Real-time Demo
To launch the real-time websocket demo, navigate to the `websocket_har_demo` directory and follow its internal `README.md` for starting the backend server and frontend dashboard.

## Dependencies

- Python 3.8+ (TensorFlow, Keras, NumPy, Pandas, Scikit-Learn, Matplotlib)
- RISC-V Toolchain (e.g., `riscv64-unknown-elf-gcc`)
- Spike RISC-V Simulator
- TensorFlow Lite for Microcontrollers (TFLite Micro)
