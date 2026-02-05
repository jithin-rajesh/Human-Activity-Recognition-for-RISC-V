"""
1D CNN for Human Activity Recognition using UCI HAR Dataset
Floating-point training with TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

def build_cnn_model(input_shape, num_classes):
    """Build 1D CNN model for HAR"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv1D(64, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Second Conv Block
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Third Conv Block
        layers.Conv1D(256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

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
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
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
    print("TensorFlow Configuration")
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
    print("\nBuilding 1D CNN model...")
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = train_model(model, X_train, y_train, X_test, y_test)
    
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
    
    # Save model
    model_path = 'models/har_cnn_float32.keras'
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, history

if __name__ == '__main__':
    model, history = main()
