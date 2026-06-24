"""
server.py — WebSocket HAR Demo Server
- Loads both CNN and CNN-GRU models with their scalers
- Applies proper normalization during inference
- Throttles predictions (every 10 samples, not every 1)
- Broadcasts to all clients (phone + laptop dashboard)
- Serves /dashboard for laptop viewing
"""
import os
import csv
import json
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import socketio
import uvicorn

# --- Configuration ---
DATA_DIR = "data"
MODELS_DIR = "models"
WINDOW_SIZE = 128
CHANNELS = 6
ACTIVITIES = ["walking", "running", "sitting", "standing"]
PREDICT_EVERY = 10  # Only predict every N new samples (throttle)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load Models + Scalers ---
def load_model_and_scaler(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.json")
    model, scaler = None, None
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"  Loaded model: {model_path}")
        if os.path.exists(scaler_path):
            with open(scaler_path) as f:
                scaler = json.load(f)
            print(f"  Loaded scaler: {scaler_path}")
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
    return model, scaler

print("Loading models...")
cnn_model, cnn_scaler = load_model_and_scaler("cnn")
gru_model, gru_scaler = load_model_and_scaler("cnn_gru")

# Also try the old demo_model as fallback
if cnn_model is None:
    old_path = os.path.join(MODELS_DIR, "demo_model.keras")
    if os.path.exists(old_path):
        cnn_model = tf.keras.models.load_model(old_path)
        print(f"  Loaded fallback: {old_path}")

# --- State ---
window_buffer = []
sample_count = 0

# --- Apps ---
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard")
async def dashboard():
    with open("dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

def normalize_window(window_np, scaler):
    """Apply saved scaler parameters to a window."""
    if scaler is None:
        return window_np
    result = window_np.copy()
    for c in range(CHANNELS):
        params = scaler.get(str(c), {"mean": 0, "std": 1})
        result[:, c] = (result[:, c] - params["mean"]) / params["std"]
    return result

def run_inference(window_np, model, scaler):
    """Normalize + predict with one model."""
    normalized = normalize_window(window_np, scaler)
    input_data = normalized.reshape(1, WINDOW_SIZE, CHANNELS)
    preds = model.predict(input_data, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    return ACTIVITIES[idx], conf

@sio.event
async def sensor_data(sid, data):
    global window_buffer, sample_count

    sample = [
        data.get('ax', 0), data.get('ay', 0), data.get('az', 0),
        data.get('gx', 0), data.get('gy', 0), data.get('gz', 0)
    ]
    label = data.get('label')

    # --- Collection Mode ---
    if label:
        filename = os.path.join(DATA_DIR, f"{label}.csv")
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['ts', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
            writer.writerow([data.get('ts'), *sample])
        return

    # --- Inference Mode ---
    window_buffer.append(sample)
    if len(window_buffer) > WINDOW_SIZE:
        window_buffer.pop(0)
    sample_count += 1

    # Only predict every PREDICT_EVERY samples to avoid spam
    if len(window_buffer) < WINDOW_SIZE:
        return
    if sample_count % PREDICT_EVERY != 0:
        return

    window_np = np.array(window_buffer, dtype=np.float32)
    results = {}

    if cnn_model is not None:
        label_cnn, conf_cnn = run_inference(window_np, cnn_model, cnn_scaler)
        results['cnn'] = {'label': label_cnn, 'confidence': conf_cnn}

    if gru_model is not None:
        label_gru, conf_gru = run_inference(window_np, gru_model, gru_scaler)
        results['cnn_gru'] = {'label': label_gru, 'confidence': conf_gru}

    if results:
        # Pick the best one for the main prediction label
        best = max(results.items(), key=lambda x: x[1]['confidence'])
        payload = {
            'label': best[1]['label'],
            'confidence': best[1]['confidence'],
            'models': results
        }
        print(f"[{best[0].upper()}] {best[1]['label']:10s} ({best[1]['confidence']*100:.0f}%)"
              + (f"  |  CNN: {results.get('cnn',{}).get('label','—')} "
                 f"GRU: {results.get('cnn_gru',{}).get('label','—')}" if len(results) > 1 else ""))
        await sio.emit('prediction', payload)

if __name__ == "__main__":
    print(f"\nStarting server on http://0.0.0.0:8000")
    print(f"  Phone:   http://<ngrok-url>/")
    print(f"  Laptop:  http://localhost:8000/dashboard\n")
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
