import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "demo_model.keras")
WINDOW_SIZE = 128
CHANNELS = 6
ACTIVITIES = ["walking", "running", "sitting", "standing"]
LABEL_MAP = {act: i for i, act in enumerate(ACTIVITIES)}

def load_and_preprocess():
    X, y = [], []
    
    for activity in ACTIVITIES:
        csv_path = os.path.join(DATA_DIR, f"{activity}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        print(f"Processing {activity}...")
        df = pd.read_csv(csv_path)
        data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
        
        # Segment into windows
        for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE // 2): # 50% overlap for more data
            window = data[i:i + WINDOW_SIZE]
            X.append(window)
            y.append(LABEL_MAP[activity])
            
    if not X:
        raise ValueError("No data found in directory. Record some samples using server.py first!")
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Scaling (simple per-channel normalization)
    # In a real app, you'd save this scaler for live inference
    for c in range(CHANNELS):
        mean = X[:, :, c].mean()
        std = X[:, :, c].std() + 1e-7
        X[:, :, c] = (X[:, :, c] - mean) / std
        
    return X, y

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, CHANNELS)),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(ACTIVITIES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("Loading data...")
    try:
        X, y = load_and_preprocess()
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Dataset: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} channels")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building model...")
    model = build_model()
    model.summary()
    
    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        verbose=1
    )
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Done! Restart server.py to use the new model.")

if __name__ == "__main__":
    main()
