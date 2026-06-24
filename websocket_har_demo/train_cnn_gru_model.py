"""
train_cnn_gru_model.py - CNN-GRU Hybrid for HAR demo
Captures both spatial (CNN) and temporal (GRU) patterns.
Saves model AND scaler parameters for consistent inference.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_gru_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "cnn_gru_scaler.json")
WINDOW_SIZE = 128
CHANNELS = 6
ACTIVITIES = ["walking", "running", "sitting", "standing"]
LABEL_MAP = {act: i for i, act in enumerate(ACTIVITIES)}

def load_and_preprocess():
    X, y = [], []
    for activity in ACTIVITIES:
        csv_path = os.path.join(DATA_DIR, f"{activity}.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found. Skipping.")
            continue
        df = pd.read_csv(csv_path)
        data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values.astype(np.float32)
        n_windows = 0
        for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE // 2):
            X.append(data[i:i + WINDOW_SIZE])
            y.append(LABEL_MAP[activity])
            n_windows += 1
        print(f"  {activity:12s}: {n_windows} windows ({len(data)} samples)")
    if not X:
        raise ValueError("No data found! Record samples first.")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    scaler_params = {}
    for c in range(CHANNELS):
        mean = float(X[:, :, c].mean())
        std = float(X[:, :, c].std()) + 1e-7
        X[:, :, c] = (X[:, :, c] - mean) / std
        scaler_params[str(c)] = {"mean": mean, "std": std}
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SCALER_PATH, 'w') as f:
        json.dump(scaler_params, f)
    print(f"  Scaler saved to {SCALER_PATH}")
    return X, y

def build_cnn_gru():
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, CHANNELS)),
        # CNN feature extraction
        layers.Conv1D(32, 5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(64, 5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        # GRU temporal modeling
        layers.GRU(64, return_sequences=False),
        layers.Dropout(0.3),
        # Head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(ACTIVITIES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("=" * 50)
    print("Training CNN-GRU Model")
    print("=" * 50)
    X, y = load_and_preprocess()
    print(f"\nDataset: {X.shape[0]} windows × {WINDOW_SIZE} steps × {CHANNELS} ch")
    print(f"Classes: {np.bincount(y)} (walk/run/sit/stand)\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_cnn_gru()
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=16, callbacks=callbacks, verbose=1)
    
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nCNN-GRU Validation Accuracy: {acc*100:.1f}%")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
