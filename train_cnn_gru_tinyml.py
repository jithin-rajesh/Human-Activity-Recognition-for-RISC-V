"""
CNN-GRU with Attention for Human Activity Recognition (TinyML/RISC-V Optimized)
Based on: "Hybrid CNN-LSTM-GRU with Attention for Human Activity Recognition" (Dangol et al., 2025)
Modified for resource-constrained RISC-V microcontrollers:
  - Reduced CNN filters (128 -> 64)
  - Removed LSTM layer (memory savings)
  - Smaller dense layers (128 -> 64)
  - Kept Residual Connections and Attention for accuracy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D,
    Bidirectional, GRU, Dense, Add, GaussianNoise, ReLU
)
import os

# Activity labels for UCI HAR dataset
ACTIVITY_LABELS = {
    0: 'Walking',
    1: 'Walking Upstairs',
    2: 'Walking Downstairs',
    3: 'Sitting',
    4: 'Standing',
    5: 'Laying'
}


class SelfAttention(layers.Layer):
    """
    Self-Attention Layer for sequence data (TinyML optimized).
    Computes attention weights and applies them to create a context vector.
    """
    
    def __init__(self, units=32, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch, timesteps, features)
        
        # Compute attention scores
        # (batch, timesteps, features) @ (features, units) -> (batch, timesteps, units)
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[-1], [0]]) + self.b)
        
        # (batch, timesteps, units) @ (units,) -> (batch, timesteps)
        attention_weights = tf.tensordot(score, self.u, axes=[[-1], [0]])
        
        # Apply softmax to get normalized attention weights
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # Expand dims for broadcasting: (batch, timesteps) -> (batch, timesteps, 1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        # Apply attention weights to input
        # (batch, timesteps, features) * (batch, timesteps, 1) -> (batch, timesteps, features)
        weighted = x * attention_weights
        
        # Sum over timesteps to get context vector: (batch, features)
        context = tf.reduce_sum(weighted, axis=1)
        
        return context
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({'units': self.units})
        return config


def build_cnn_gru_attention_tinyml(input_shape, num_classes):
    """
    Build CNN-GRU with Attention model optimized for TinyML/RISC-V.
    
    Architecture (Modified from Dangol et al., 2025 for resource constraints):
    1. Input with GaussianNoise (robustness)
    2. CNN Block with Residual Connection (64 filters - reduced)
    3. Bidirectional GRU only (NO LSTM - memory savings)
    4. Self-Attention
    5. Compact Dense Output (64 units)
    
    Args:
        input_shape: Tuple of (n_timesteps, n_features)
        num_classes: Number of output classes
        
    Returns:
        Keras Model
    """
    
    # ========== INPUT LAYER ==========
    inputs = Input(shape=input_shape, name='input')
    
    # Add Gaussian noise for robustness during training
    x = GaussianNoise(0.1, name='gaussian_noise')(inputs)
    
    # ========== CNN BLOCK WITH RESIDUAL CONNECTION ==========
    # First Conv1D layer (64 filters for TinyML efficiency)
    conv1 = Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        name='conv1d_1'
    )(x)
    conv1 = BatchNormalization(name='bn_1')(conv1)
    conv1 = ReLU(name='relu_1')(conv1)
    
    # Second Conv1D layer
    conv2 = Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        name='conv1d_2'
    )(conv1)
    conv2 = BatchNormalization(name='bn_2')(conv2)
    conv2 = ReLU(name='relu_2')(conv2)
    conv2 = Dropout(0.3, name='dropout_cnn')(conv2)
    
    # Residual Connection: 1x1 Conv to match dimensions (9 -> 64 channels)
    residual = Conv1D(
        filters=64,
        kernel_size=1,
        padding='same',
        name='residual_projection'
    )(x)
    
    # Add residual connection
    cnn_output = Add(name='residual_add')([conv2, residual])
    
    # MaxPooling after residual addition
    cnn_output = MaxPooling1D(pool_size=2, name='maxpool')(cnn_output)
    
    # ========== RECURRENT BLOCK (GRU ONLY - NO LSTM) ==========
    # Bidirectional GRU (memory-efficient alternative to LSTM)
    gru_out = Bidirectional(
        GRU(
            units=64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='gru'
        ),
        name='bidirectional_gru'
    )(cnn_output)
    gru_out = Dropout(0.3, name='dropout_gru')(gru_out)
    
    # ========== ATTENTION MECHANISM ==========
    # Apply self-attention to get context vector
    attention_output = SelfAttention(units=32, name='self_attention')(gru_out)
    
    # ========== OUTPUT LAYERS (COMPACT) ==========
    # Dense layer (reduced from 128 to 64)
    dense = Dense(64, activation='relu', name='dense_1')(attention_output)
    dense = Dropout(0.3, name='dropout_dense')(dense)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='output')(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CNN_GRU_Attention_TinyML')
    
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


def load_data():
    """Load preprocessed UCI HAR dataset"""
    X_train = np.load('assets/X_train_raw_scaled.npy')
    X_test = np.load('assets/X_test_raw_scaled.npy')
    y_train = np.load('assets/y_train_one_hot.npy')
    y_test = np.load('assets/y_test_one_hot.npy')
    
    # Convert to float32 for GPU efficiency
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"Training data: {X_train.shape} -> {y_train.shape}")
    print(f"Test data: {X_test.shape} -> {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
    """Train the model"""
    
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
            'models/har_cnn_gru_attention_tinyml_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
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
    print("\nConfusion Matrix Summary:")
    print("-" * 40)
    try:
        from sklearn.metrics import confusion_matrix, classification_report
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


def main():
    # Check GPU availability
    print("=" * 60)
    print("CNN-GRU with Attention for HAR (TinyML/RISC-V Optimized)")
    print("Based on Dangol et al., 2025 (Modified for Edge Deployment)")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    print("=" * 60 + "\n")
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Model parameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # (128, 9)
    num_classes = y_train.shape[1]  # 6
    
    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Build model
    print("\nBuilding CNN-GRU-Attention model (TinyML optimized)...")
    model = build_cnn_gru_attention_tinyml(input_shape, num_classes)
    model.summary()
    
    # Print model size analysis
    size_kb, size_mb, total_params = estimate_model_size(model)
    trainable, non_trainable, _ = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("TinyML Model Size Analysis:")
    print("=" * 60)
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable:,}")
    print(f"  Non-trainable:         {non_trainable:,}")
    print(f"  Model Size (float32):  {size_kb:.2f} KB ({size_mb:.2f} MB)")
    print(f"  Model Size (int8 est): {size_kb/4:.2f} KB (quantized)")
    print("=" * 60)
    
    # Print architecture details
    print("\nArchitecture Summary (TinyML Optimized):")
    print("=" * 60)
    print("1. Input + GaussianNoise (robustness)")
    print("2. CNN Block (2x Conv1D-64) with Residual Connection")
    print("3. Bidirectional GRU (64 units) - NO LSTM (memory savings)")
    print("4. Self-Attention Layer")
    print("5. Dense (64) -> Output (6 classes)")
    print("=" * 60)
    print("\nOptimizations for RISC-V:")
    print("  ✓ Reduced CNN filters: 128 -> 64")
    print("  ✓ Removed LSTM layer (saves ~50% RNN memory)")
    print("  ✓ Smaller dense layer: 128 -> 64")
    print("  ✓ Compact attention: 32 units")
    print("=" * 60)
    
    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    evaluate_model(model, X_test, y_test)
    
    # Test some individual predictions
    print("\n" + "=" * 60)
    print("Testing Individual Predictions")
    print("=" * 60)
    
    # Test 5 random samples
    np.random.seed(42)
    test_indices = np.random.choice(len(X_test), 5, replace=False)
    correct = 0
    for idx in test_indices:
        if test_single_prediction(model, X_test, y_test, idx):
            correct += 1
    
    print(f"\n{'='*50}")
    print(f"Random sample test: {correct}/5 correct")
    print(f"{'='*50}")
    
    # Save final model
    model_path = 'models/har_cnn_gru_attention_tinyml_final.keras'
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"\nFinal model saved to: {model_path}")
    
    # Also save in SavedModel format for TFLite conversion
    saved_model_path = 'models/har_cnn_gru_attention_tinyml_savedmodel'
    model.save(saved_model_path)
    print(f"SavedModel format saved to: {saved_model_path}")
    
    # Convert to TFLite (float32)
    print("\n" + "=" * 60)
    print("Converting to TensorFlow Lite...")
    print("=" * 60)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = 'models/har_cnn_gru_attention_tinyml.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        tflite_size_kb = len(tflite_model) / 1024
        print(f"TFLite model saved to: {tflite_path}")
        print(f"TFLite model size: {tflite_size_kb:.2f} KB")
        
        # Convert to TFLite with int8 quantization
        print("\nConverting to TFLite (INT8 Quantized)...")
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization
        def representative_dataset():
            for i in range(min(100, len(X_train))):
                yield [X_train[i:i+1]]
        
        converter_int8.representative_dataset = representative_dataset
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        
        tflite_int8_model = converter_int8.convert()
        tflite_int8_path = 'models/har_cnn_gru_attention_tinyml_int8.tflite'
        with open(tflite_int8_path, 'wb') as f:
            f.write(tflite_int8_model)
        tflite_int8_size_kb = len(tflite_int8_model) / 1024
        print(f"TFLite INT8 model saved to: {tflite_int8_path}")
        print(f"TFLite INT8 model size: {tflite_int8_size_kb:.2f} KB")
        
        print("\n" + "=" * 60)
        print("Model Size Comparison:")
        print("=" * 60)
        print(f"  Keras (float32):    {size_kb:.2f} KB")
        print(f"  TFLite (float32):   {tflite_size_kb:.2f} KB")
        print(f"  TFLite (int8):      {tflite_int8_size_kb:.2f} KB")
        print(f"  Compression ratio:  {size_kb/tflite_int8_size_kb:.1f}x")
        print("=" * 60)
        
    except Exception as e:
        print(f"TFLite conversion error: {e}")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy (TinyML)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss (TinyML)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history_tinyml.png', dpi=150)
        print("\nTraining history plot saved to: models/training_history_tinyml.png")
        plt.show()
    except ImportError:
        print("(matplotlib not available for plotting)")
    
    return model, history


if __name__ == '__main__':
    model, history = main()
