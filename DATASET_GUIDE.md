# RealWorld HAR Dataset — Download, Extraction & Preprocessing Guide

This guide walks through obtaining and preparing the **RealWorld Human Activity Recognition** dataset for training HAR models with proper subject-aware (LOSO) evaluation.

---

## Overview

| Property | Value |
|---|---|
| **Source** | University of Mannheim |
| **URL** | https://sensor.informatik.uni-mannheim.de/#dataset_realworld |
| **Subjects** | 15 (labelled `proband1` – `proband15`) |
| **Activities** | 8: climbing down, climbing up, jumping, lying, running, sitting, standing, walking |
| **Sensors** | Accelerometer, Gyroscope, Magnetometer (3 axes each = 9 channels) |
| **Body Positions** | 7: chest, forearm, head, shin, thigh, upper arm, waist |
| **Sample Rate** | ~50 Hz |
| **Download Size** | ~1 GB (for the 3 sensors we need) |

---

## Step 1: Download the Raw Data

1. Go to: **https://sensor.informatik.uni-mannheim.de/#dataset_realworld**
2. Download **all 15 subjects** (`proband1` through `proband15`)
   - Each subject is a separate zip file
3. Save them to `rawdata/` inside the project directory:

```
Human-Activity-Recognition-for-RISC-V/
  rawdata/
    proband1.zip
    proband2.zip
    ...
    proband15.zip
```

---

## Step 2: Extract the Zip Files

Each subject zip contains further nested zips organised by sensor and activity. We only need **3 sensor types** (CSV versions):

| Prefix | Sensor | Channels |
|---|---|---|
| `acc_*_csv.zip` | Accelerometer | x, y, z |
| `gyr_*_csv.zip` | Gyroscope | x, y, z |
| `mag_*_csv.zip` | Magnetometer | x, y, z |

**Skip these** (not needed):
- `gps_*` — GPS coordinates
- `lig_*` — Light sensor
- `mic_*` — Microphone / sound level
- `*_sqlite.zip` — Same data in SQLite format (redundant)

### Extraction Commands

```bash
cd rawdata

# 1. Unzip each subject's main zip
for i in $(seq 1 15); do
    unzip -o "proband${i}.zip" -d .
done

# 2. For each subject, extract only the needed sensor CSVs
for i in $(seq 1 15); do
    echo "Extracting sensor CSVs for proband${i}..."
    cd "proband${i}/data"

    # Extract accelerometer, gyroscope, and magnetometer CSVs
    for f in acc_*_csv.zip gyr_*_csv.zip mag_*_csv.zip; do
        [ -f "$f" ] && unzip -o "$f"
    done

    cd ../..
done
```

### Expected File Structure After Extraction

```
rawdata/
  proband1/
    data/
      acc_climbingdown_chest.csv
      acc_climbingdown_forearm.csv
      acc_climbingdown_head.csv
      acc_climbingdown_shin.csv
      acc_climbingdown_thigh.csv
      acc_climbingdown_upperarm.csv
      acc_climbingdown_waist.csv
      acc_climbingup_chest.csv
      ...
      gyr_walking_waist.csv
      mag_standing_shin.csv
      ...
  proband2/
    data/
      ...
  ...
  proband15/
    data/
      ...
```

Each CSV file has this format:
```
attr_time,attr_x,attr_y,attr_z
1476710000000,0.123,-9.81,0.456
1476710020000,0.125,-9.79,0.461
...
```

- `attr_time` — Unix timestamp in nanoseconds
- `attr_x, attr_y, attr_z` — Sensor readings for the 3 axes

---

## Step 3: Write and Run `preprocess_realworld.py`

Create `preprocess_realworld.py` in the project root. Below is the full specification.

```bash
cd Human-Activity-Recognition-for-RISC-V
python preprocess_realworld.py
```

---

### 3.1 Configuration (Constants at the Top)

```python
RAW_DATA_DIR   = 'rawdata'      # contains proband1/ ... proband15/
OUTPUT_DIR     = 'assets'
BODY_POSITION  = 'chest'        # one of: chest, forearm, head, shin, thigh, upperarm, waist
WINDOW_SIZE    = 128             # samples per window (128 @ 50 Hz = 2.56 s)
WINDOW_OVERLAP = 0              # 0 = no overlap (safest for LOSO, avoids leakage)
SAMPLE_RATE    = 50              # Hz
SENSORS        = ['acc', 'gyr', 'mag']   # 3 axes each = 9 channels
NUM_SUBJECTS   = 15
```

---

### 3.2 Activity Label Mapping

Map activity folder/file names to integer IDs. Must match the labels used in training.

```python
ACTIVITIES = {
    'climbingdown': 0,
    'climbingup':   1,
    'jumping':      2,
    'lying':        3,
    'running':      4,
    'sitting':      5,
    'standing':     6,
    'walking':      7,
}
NUM_CLASSES = 8
```

---

### 3.3 CSV Parsing

**Input:** A single CSV file, e.g. `rawdata/proband1/data/acc_walking_chest.csv`

**Format:**
```
attr_time,attr_x,attr_y,attr_z
1476710000000,0.123,-9.81,0.456
```

**Steps:**
1. Read with `numpy.genfromtxt(path, delimiter=',', skip_header=1)` (or pandas)
2. Column 0 = `attr_time` (nanosecond Unix timestamp)
3. Columns 1-3 = `attr_x, attr_y, attr_z` (sensor readings, float32)
4. Drop any rows containing NaN values
5. If fewer than `WINDOW_SIZE` valid rows remain, skip this file entirely
6. Return: `timestamps` array (N,) and `values` array (N, 3)

---

### 3.4 Sensor Alignment

The 3 sensors (acc, gyr, mag) are recorded **independently** with different timestamps. They must be aligned before stacking.

**Method: nearest-neighbor interpolation, using accelerometer as reference**

1. Use the accelerometer's timestamps as the reference timeline
2. For each gyroscope & magnetometer timestamp array, find the nearest match to each accelerometer timestamp using `np.searchsorted` + boundary check
3. Index the gyr/mag values by those nearest indices
4. Stack horizontally: `[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z]` → shape `(N, 9)`
5. If any of the 3 sensors is missing for a given activity, **skip that activity entirely** for this subject

---

### 3.5 Windowing (Segmentation)

Take the aligned `(N, 9)` signal and chop it into fixed-length windows.

```
step = WINDOW_SIZE - WINDOW_OVERLAP   # with overlap=0, step = 128
for start in range(0, len(data) - WINDOW_SIZE + 1, step):
    window = data[start : start + WINDOW_SIZE]   # shape (128, 9)
```

Each window gets:
- The **activity label** (integer) from the filename
- The **subject ID** (integer 1–15) from the folder name

---

### 3.6 Processing Loop (Main Logic)

```
for each subject (1 to 15):
    for each activity in ACTIVITIES:
        for each sensor in [acc, gyr, mag]:
            csv_path = rawdata/proband{id}/data/{sensor}_{activity}_{BODY_POSITION}.csv
            read CSV → (timestamps, values)

        align 3 sensors → (N, 9)
        segment into windows → list of (128, 9) arrays
        record activity label and subject ID for each window

stack all windows → X, y, subjects
```

---

### 3.7 Scaling

Apply `sklearn.preprocessing.StandardScaler` (zero mean, unit variance) **per channel** across all windows.

```python
X_flat = X.reshape(-1, 9)            # flatten windows into rows
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X = X_flat.reshape(-1, 128, 9)       # reshape back
```

> **Note:** This fits the scaler on all subjects. During LOSO training, the training script should re-fit the scaler on train-fold subjects only. This global scaling is just for storage convenience.

---

### 3.8 One-Hot Encoding

Convert integer labels to one-hot vectors:

```python
y_onehot = np.zeros((N, NUM_CLASSES), dtype=np.float32)
y_onehot[np.arange(N), y_integers] = 1.0
```

---

### 3.9 Output Files (saved to `assets/`)

| File | Shape | Dtype | Description |
|---|---|---|---|
| `X_realworld_loso.npy` | `(N, 128, 9)` | float32 | Scaled sensor windows: `[acc_xyz, gyr_xyz, mag_xyz]` |
| `y_realworld_loso.npy` | `(N, 8)` | float32 | One-hot encoded activity labels |
| `subjects_realworld_loso.npy` | `(N,)` | int32 | Subject ID (1–15) per window |

---

### 3.10 Edge Cases to Handle

| Situation | How to Handle |
|---|---|
| CSV file missing (e.g., a subject didn't do "jumping") | Skip that activity for that subject, print a warning |
| CSV has fewer rows than `WINDOW_SIZE` | Skip that file |
| Rows contain NaN | Drop those rows before windowing |
| A sensor CSV exists but is empty or corrupt | Skip, print warning |
| Not all 3 sensors available for an activity | Skip that activity (need all 3 for 9-channel alignment) |

---

### 3.11 Expected Console Output

The script should print progress so you can verify it's working:

```
[Subject  1] Processing...
    climbingdown        :   45 windows (5800 samples, 116.0s)
    climbingup          :   52 windows (6700 samples, 134.0s)
    jumping             :    8 windows (1100 samples, 22.0s)
    lying               :   93 windows (12000 samples, 240.0s)
    running             :   78 windows (10100 samples, 202.0s)
    sitting             :   91 windows (11700 samples, 234.0s)
    standing            :   88 windows (11300 samples, 226.0s)
    walking             :   82 windows (10600 samples, 212.0s)
  => 537 total windows

[Subject  2] Processing...
    ...

PREPROCESSING COMPLETE
  Total windows:  ~7000-10000
  X shape:        (N, 128, 9)
  Subjects:       [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

  Samples per subject:
    Subject  1:   537 windows
    Subject  2:   ...
    ...

  Samples per activity:
    climbingdown        :   XXX windows
    ...
```

---

### 3.12 Dependencies

```bash
pip install numpy scikit-learn
# pandas is optional but makes CSV reading easier
```

---

## Step 4: Verify the Data

```bash
python -c "
import numpy as np
X = np.load('assets/X_realworld_loso.npy')
y = np.load('assets/y_realworld_loso.npy')
s = np.load('assets/subjects_realworld_loso.npy')

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'Subjects: {np.unique(s)}')
print(f'Samples per subject:')
for subj in np.unique(s):
    print(f'  Subject {subj:2d}: {(s == subj).sum():5d} windows')
print(f'Activities: {np.unique(np.argmax(y, axis=1))}')
"
```

Expected output should show:
- ~6000–10000 total windows
- 15 unique subjects
- 8 unique activity classes
- Varying samples per subject (some subjects have more data)

---

## Step 5: Train with LOSO Evaluation

Once preprocessing is done, use the LOSO training script (provided separately) which:

1. **Holds out 2–3 subjects** as the test set
2. **Uses 1 subject** from the remaining as validation (for early stopping)
3. **Trains on the remaining 12** subjects
4. Repeats across folds and averages accuracy

This gives an **honest generalisation accuracy** — how well the model performs on people it has **never seen**.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Missing CSVs for some activities | Not all subjects did all activities (e.g., some skipped jumping). The script handles this gracefully |
| `attr_time` column missing | Some CSVs may use different column names. Check the header row |
| Very few samples for "jumping" | Normal — jumping recordings are short (~10–20 seconds vs minutes for walking) |
| Memory errors | Process one subject at a time. The script does this by default |

---

## Notes

- The original preprocessed `X_realworldhar_raw_scaled.npy` (9634 windows) did NOT preserve subject IDs, making proper LOSO evaluation impossible. That's why we reprocess from raw.
- The **body position matters significantly**. Chest gives the best overall accuracy; forearm/waist gives more realistic smartwatch/phone results but may be ~5–10% lower.
- Window overlap is intentionally set to **0%** for LOSO evaluation to avoid data leakage between adjacent windows that belong to the same recording.
