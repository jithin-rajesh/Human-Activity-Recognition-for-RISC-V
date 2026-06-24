"""
Lightweight 1D CNN for Human Activity Recognition (TinyML/RISC-V Optimized)
Trained on the RealWorld HAR Dataset (preprocessed)
Designed for resource-constrained microcontrollers with minimal memory footprint.

Optimizations:
  - Reduced filter counts: 32 -> 64 (instead of 64 -> 128 -> 256)
  - Only 2 conv blocks (sufficient for HAR)
  - Smaller dense layer (32 units)
  - GlobalAveragePooling for parameter efficiency
  - Keras Functional API for TFLite compatibility

Dataset:
  - RealWorld HAR: 9634 samples, 128 timesteps, 9 features, 8 classes
  - Single file (train/test split handled internally via sklearn)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D,
    Dense, GlobalAveragePooling1D, ReLU
)
import os

# Activity labels for RealWorld HAR dataset (8 classes)
ACTIVITY_LABELS = {
    0: 'Climbing Down',
    1: 'Climbing Up',
    2: 'Jumping',
    3: 'Lying',
    4: 'Running',
    5: 'Sitting',
    6: 'Standing',
    7: 'Walking'
}


def build_cnn_tinyml(input_shape, num_classes, filters=(32, 64), dense_units=32):
    """
    Build lightweight 1D CNN model for HAR optimized for TinyML/RISC-V.
    
    Architecture:
    1. Input Layer
    2. Conv Block 1: Conv1D(32) -> BN -> ReLU -> MaxPool
    3. Conv Block 2: Conv1D(64) -> BN -> ReLU -> MaxPool
    4. GlobalAveragePooling1D (parameter-efficient)
    5. Dense(32) -> Dropout -> Softmax
    
    Args:
        input_shape: Tuple of (n_timesteps, n_features)
        num_classes: Number of output classes
        filters: Tuple of filter counts for each conv block
        dense_units: Number of units in hidden dense layer
        
    Returns:
        Keras Model
    """
    
    # ========== INPUT LAYER ==========
    inputs = Input(shape=input_shape, name='input')
    
    # ========== CONV BLOCK 1 ==========
    x = Conv1D(
        filters=filters[0],
        kernel_size=5,
        padding='same',
        name='conv1d_1'
    )(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = ReLU(name='relu_1')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)
    
    # ========== CONV BLOCK 2 ==========
    x = Conv1D(
        filters=filters[1],
        kernel_size=5,
        padding='same',
        name='conv1d_2'
    )(x)
    x = BatchNormalization(name='bn_2')(x)
    x = ReLU(name='relu_2')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    
    # ========== GLOBAL POOLING ==========
    # GlobalAveragePooling drastically reduces parameters
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # ========== CLASSIFICATION HEAD ==========
    x = Dense(dense_units, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_dense')(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CNN_TinyML_RealWorld')
    
    return model


def count_parameters(model):
    """Count trainable and non-trainable parameters"""
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total = trainable + non_trainable
    return trainable, non_trainable, total


def estimate_model_size(model):
    """Estimate model size in memory (float32)"""
    trainable, non_trainable, total = count_parameters(model)
    size_bytes = total * 4  # float32 = 4 bytes
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return size_kb, size_mb, total


def load_data(val_size=0.15, test_size=0.15, random_state=42):
    """
    Load preprocessed RealWorld HAR dataset and split into train/val/test.

    Uses a stratified 3-way split so that:
      - val set drives EarlyStopping / ModelCheckpoint during training
      - test set is NEVER seen during training and gives an unbiased accuracy
    """
    print("Loading RealWorld HAR data...")
    X = np.load('assets/X_realworldhar_raw_scaled.npy')
    y = np.load('assets/y_realworldhar_one_hot.npy')

    # Convert to float32 for efficiency
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    y_classes = np.argmax(y, axis=1)
    np.random.seed(random_state)

    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []

    for c in range(y.shape[1]):
        indices = np.where(y_classes == c)[0]
        np.random.shuffle(indices)
        n = len(indices)
        n_val  = max(1, int(n * val_size))
        n_test = max(1, int(n * test_size))

        test_idx  = indices[:n_test]
        val_idx   = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        X_train_list.append(X[train_idx])
        X_val_list.append(X[val_idx])
        X_test_list.append(X[test_idx])
        y_train_list.append(y[train_idx])
        y_val_list.append(y[val_idx])
        y_test_list.append(y[test_idx])

    def _shuffle_pair(X_list, y_list):
        """Stack X and y, shuffle with the SAME permutation, then return both."""
        X_arr = np.vstack(X_list)
        y_arr = np.vstack(y_list)
        perm = np.random.permutation(len(X_arr))
        return X_arr[perm], y_arr[perm]

    X_train, y_train = _shuffle_pair(X_train_list, y_train_list)
    X_val,   y_val   = _shuffle_pair(X_val_list,   y_val_list)
    X_test,  y_test  = _shuffle_pair(X_test_list,  y_test_list)

    print(f"Total data: {X.shape}")
    print(f"Train data: {X_train.shape}  Val data: {X_val.shape}  Test data: {X_test.shape}")

    # Show class distribution
    print("\nClass distribution (Train / Val / Test):")
    for i, label in ACTIVITY_LABELS.items():
        tr = (np.argmax(y_train, axis=1) == i).sum()
        va = (np.argmax(y_val,   axis=1) == i).sum()
        te = (np.argmax(y_test,  axis=1) == i).sum()
        print(f"  {label:20s}: {tr:5d} / {va:4d} / {te:4d}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    """
    Train the model.

    X_val / y_val are used ONLY for EarlyStopping and ModelCheckpoint.
    The held-out test set must NOT be passed here.
    """

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/har_realworld_cnn_tinyml_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Train (validation set drives early stopping — test set never touched here)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model and show per-class accuracy"""
    
    # Overall accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*50}\n")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Per-class accuracy
    print("Per-class accuracy:")
    print("-" * 40)
    for i, label in ACTIVITY_LABELS.items():
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean() * 100
            print(f"  {label:20s}: {class_acc:6.2f}% ({mask.sum()} samples)")
    
    # Confusion matrix summary
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        print("\nConfusion Matrix:")
        print("-" * 40)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                    target_names=list(ACTIVITY_LABELS.values())))
    except ImportError:
        print("(sklearn not available for detailed metrics)")
    
    return y_pred_classes, y_true_classes


def test_single_prediction(model, X_test, y_test, sample_idx=0):
    """Test prediction on a single sample"""
    sample = X_test[sample_idx:sample_idx+1]
    true_label = np.argmax(y_test[sample_idx])
    
    prediction = model.predict(sample, verbose=0)
    pred_label = np.argmax(prediction[0])
    confidence = prediction[0][pred_label] * 100
    
    print(f"\n{'='*50}")
    print("Single Sample Prediction Test")
    print(f"{'='*50}")
    print(f"True Activity:      {ACTIVITY_LABELS[true_label]}")
    print(f"Predicted Activity: {ACTIVITY_LABELS[pred_label]}")
    print(f"Confidence:         {confidence:.2f}%")
    print(f"Match:              {'✓ CORRECT' if pred_label == true_label else '✗ INCORRECT'}")
    
    print("\nAll class probabilities:")
    for i, prob in enumerate(prediction[0]):
        marker = " <--" if i == pred_label else ""
        print(f"  {ACTIVITY_LABELS[i]:20s}: {prob*100:6.2f}%{marker}")
    
    return pred_label == true_label


def convert_to_tflite(model, X_train, model_name='har_realworld_cnn_tinyml'):
    """Convert model to TFLite formats (float32 and int8)"""
    
    print("\n" + "=" * 60)
    print("Converting to TensorFlow Lite...")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    results = {}
    
    # Float32 TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = f'models/{model_name}.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        results['float32'] = len(tflite_model) / 1024
        print(f"✓ TFLite (float32): {tflite_path} ({results['float32']:.2f} KB)")
    except Exception as e:
        print(f"✗ TFLite float32 conversion failed: {e}")
    
    # Int8 Quantized TFLite
    try:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization calibration
        def representative_dataset():
            for i in range(min(100, len(X_train))):
                yield [X_train[i:i+1]]
        
        converter_int8.representative_dataset = representative_dataset
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        
        tflite_int8_model = converter_int8.convert()
        tflite_int8_path = f'models/{model_name}_int8.tflite'
        with open(tflite_int8_path, 'wb') as f:
            f.write(tflite_int8_model)
        results['int8'] = len(tflite_int8_model) / 1024
        print(f"✓ TFLite (int8):    {tflite_int8_path} ({results['int8']:.2f} KB)")
    except Exception as e:
        print(f"✗ TFLite int8 conversion failed: {e}")
    
    return results


def main():
    # Check GPU availability
    print("=" * 60)
    print("Lightweight 1D CNN for RealWorld HAR (TinyML/RISC-V Optimized)")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    print("=" * 60 + "\n")
    
    # Load data — three-way split to avoid test-set leakage
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Model parameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # (128, 9)
    num_classes = y_train.shape[1]  # 8
    
    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Build model
    print("\nBuilding Lightweight CNN model for RealWorld HAR...")
    model = build_cnn_tinyml(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=(32, 64),      # Reduced from (64, 128, 256)
        dense_units=32         # Reduced from 128
    )
    model.summary()
    
    # Model size analysis
    size_kb, size_mb, total_params = estimate_model_size(model)
    trainable, non_trainable, _ = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("TinyML Model Size Analysis:")
    print("=" * 60)
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable:,}")
    print(f"  Non-trainable:         {non_trainable:,}")
    print(f"  Model Size (float32):  {size_kb:.2f} KB ({size_mb:.4f} MB)")
    print(f"  Model Size (int8 est): {size_kb/4:.2f} KB (quantized)")
    print("=" * 60)
    
    # Architecture summary
    print("\nArchitecture Summary:")
    print("=" * 60)
    print("  Layer              Filters   Output Shape")
    print("  " + "-" * 50)
    print(f"  Input              -         {input_shape}")
    print(f"  Conv1D + BN + ReLU 32        ({input_shape[0]}, 32)")
    print(f"  MaxPool1D          -         ({input_shape[0]//2}, 32)")
    print(f"  Conv1D + BN + ReLU 64        ({input_shape[0]//2}, 64)")
    print(f"  MaxPool1D          -         ({input_shape[0]//4}, 64)")
    print(f"  GlobalAvgPool1D    -         (64,)")
    print(f"  Dense + Dropout    32        (32,)")
    print(f"  Output (Softmax)   {num_classes}         ({num_classes},)")
    print("=" * 60)
    
    # Train (val set used for callbacks; test set untouched)
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100)
    
    # Evaluate on the held-out test set (never seen during training)
    print("\n" + "=" * 60)
    print("Evaluation Results (held-out test set)")
    print("=" * 60)
    evaluate_model(model, X_test, y_test)
    
    # Test individual predictions
    print("\n" + "=" * 60)
    print("Testing Individual Predictions")
    print("=" * 60)
    
    np.random.seed(42)
    test_indices = np.random.choice(len(X_test), 5, replace=False)
    correct = 0
    for idx in test_indices:
        if test_single_prediction(model, X_test, y_test, idx):
            correct += 1
    
    print(f"\n{'='*50}")
    print(f"Random sample test: {correct}/5 correct")
    print(f"{'='*50}")
    
    # Save Keras model
    model_path = 'models/har_realworld_cnn_tinyml_final.keras'
    model.save(model_path)
    print(f"\nKeras model saved to: {model_path}")
    
    # Convert to TFLite
    tflite_sizes = convert_to_tflite(model, X_train, 'har_realworld_cnn_tinyml')  # X_val/X_test never used for calibration
    
    # Final size comparison
    if tflite_sizes:
        print("\n" + "=" * 60)
        print("Final Model Size Comparison:")
        print("=" * 60)
        print(f"  Keras (float32):    {size_kb:.2f} KB")
        if 'float32' in tflite_sizes:
            print(f"  TFLite (float32):   {tflite_sizes['float32']:.2f} KB")
        if 'int8' in tflite_sizes:
            print(f"  TFLite (int8):      {tflite_sizes['int8']:.2f} KB")
            print(f"  Compression ratio:  {size_kb/tflite_sizes['int8']:.1f}x")
        print("=" * 60)
        print("\n✓ INT8 model is ready for RISC-V deployment!")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy (RealWorld HAR TinyML CNN)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss (RealWorld HAR TinyML CNN)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history_realworld_cnn_tinyml.png', dpi=150)
        print("\nTraining history saved to: models/training_history_realworld_cnn_tinyml.png")
    except ImportError:
        print("(matplotlib not available for plotting)")
    
    return model, history


if __name__ == '__main__':
    model, history = main()
