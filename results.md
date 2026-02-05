# Model Evaluation Results

This document contains the evaluation metrics for all trained models on the UCI HAR dataset.

## Summary Comparison

| Model | Test Accuracy | Test Loss |
|-------|---------------|-----------|
| Standard CNN (Float32) | 94.37% | 0.2438 |
| Hybrid CNN-LSTM-GRU Attention | 93.55% | 0.2138 |
| TinyML CNN (Float32) | 93.59% | 0.2390 |
| CNN-GRU TinyML (Float32) | 93.38% | 0.2571 |

## Detailed Metrics

### 1. Standard CNN (Float32)
*Model File: `models/har_cnn_float32.keras`*

- **Test Accuracy**: 94.37%
- **Test Loss**: 0.2438

**Per-class Accuracy:**
| Activity | Accuracy | Samples |
|----------|----------|---------|
| Walking | 99.80% | 496 |
| Walking Upstairs | 95.33% | 471 |
| Walking Downstairs | 99.05% | 420 |
| Sitting | 86.56% | 491 |
| Standing | 86.28% | 532 |
| Laying | 100.00% | 537 |

### 2. Hybrid CNN-LSTM-GRU Attention
*Model File: `models/har_cnn_lstm_gru_attention_final.keras`*

- **Test Accuracy**: 93.55%
- **Test Loss**: 0.2138

**Per-class Accuracy:**
| Activity | Accuracy | Samples |
|----------|----------|---------|
| Walking | 99.80% | 496 |
| Walking Upstairs | 95.97% | 471 |
| Walking Downstairs | 99.05% | 420 |
| Sitting | 82.69% | 491 |
| Standing | 85.15% | 532 |
| Laying | 99.63% | 537 |

### 3. TinyML CNN (Float32)
*Model File: `models/har_cnn_tinyml_final.keras`*

- **Test Accuracy**: 93.59%
- **Test Loss**: 0.2390

**Per-class Accuracy:**
| Activity | Accuracy | Samples |
|----------|----------|---------|
| Walking | 99.80% | 496 |
| Walking Upstairs | 93.84% | 471 |
| Walking Downstairs | 100.00% | 420 |
| Sitting | 82.89% | 491 |
| Standing | 85.90% | 532 |
| Laying | 100.00% | 537 |

### 4. CNN-GRU TinyML (Float32)
*Model File: `models/har_cnn_gru_attention_tinyml_final.keras`*

- **Test Accuracy**: 93.38%
- **Test Loss**: 0.2571

**Per-class Accuracy:**
| Activity | Accuracy | Samples |
|----------|----------|---------|
| Walking | 97.78% | 496 |
| Walking Upstairs | 94.69% | 471 |
| Walking Downstairs | 97.86% | 420 |
| Sitting | 83.91% | 491 |
| Standing | 86.65% | 532 |
| Laying | 100.00% | 537 |
