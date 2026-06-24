# WebSocket HAR Real-time Demo

This directory contains a complete real-time Human Activity Recognition demo using an iPhone as the sensor and a Python server for inference.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python server.py
   ```

3. **Expose to HTTPS**:
   iOS requires HTTPS to access motion sensors. Use `ngrok` or similar:
   ```bash
   ngrok http 8000
   ```

## Demo Flow

### Phase 1: Data Collection (Two Options)

#### Option A: Real-time Collection (Via Browser)
1. Open the `ngrok` URL on your iPhone.
2. Tap **"Enable Motion Sensors"** and grant permission.
3. Select an activity (e.g., "Walking").
4. Tap **"Start Collection"** and walk for ~60 seconds.
5. Tap **"Stop"**.

#### Option B: Use Existing Phyphox CSVs
If you already have recordings from the Phyphox app:
1. Place your CSV files in the `websocket_har_demo` directory.
2. Ensure filenames contain the activity (e.g., `walking_data.csv`).
3. Run the converter:
   ```bash
   python convert_phyphox.py
   ```
4. This will populate the `data/` folder for you.

### Phase 2: Training
1. On your laptop, run:
   ```bash
   python train_demo_model.py
   ```
2. This will create `models/demo_model.keras`.

### Phase 3: Live Inference
1. Restart the server:
   ```bash
   python server.py
   ```
2. Refresh the page on your iPhone.
3. Your live activity will now be displayed in the large box!

---

## Directory Structure
- `server.py`: FastAPI + Socket.io server.
- `index.html`: Mobile frontend.
- `train_demo_model.py`: Script to train on your own data.
- `data/`: CSV files recorded from the phone.
- `models/`: The trained demo model.
