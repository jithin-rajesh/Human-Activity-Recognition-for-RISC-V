#!/usr/bin/env python3
"""
preprocess_realworld.py
-----------------------
Preprocess the RealWorld HAR dataset into LOSO-ready numpy arrays.

Reads raw CSVs from the extracted dataset, aligns 3 sensor streams
(accelerometer, gyroscope, magnetometer), segments into fixed-size
windows, scales, one-hot encodes labels, and saves .npy files with
subject IDs preserved.

Usage:
    python preprocess_realworld.py
"""

import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler

# ─── Configuration ──────────────────────────────────────────────────────────
RAW_DATA_DIR   = '/home/the-stick-insect/Downloads/realworldhar'
OUTPUT_DIR     = 'assets'
BODY_POSITION  = 'chest'
WINDOW_SIZE    = 128        # samples per window (128 @ 50 Hz ≈ 2.56 s)
WINDOW_OVERLAP = 0          # 0 = no overlap
SAMPLE_RATE    = 50         # Hz
NUM_SUBJECTS   = 15

# Sensor CSV prefix mapping
# Accelerometer files use: acc_{activity}_{position}.csv
# Gyroscope files use:     Gyroscope_{activity}_{position}.csv
# Magnetometer files use:  MagneticField_{activity}_{position}.csv
SENSOR_PREFIXES = {
    'acc': 'acc',
    'gyr': 'Gyroscope',
    'mag': 'MagneticField',
}

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

# ─── CSV Parsing ────────────────────────────────────────────────────────────

def find_csv(data_dir, sensor_key, activity, position):
    """
    Find the CSV file for a given sensor/activity/position.
    Handles naming quirks:
      - Some activities have a '_2' or '_3' suffix variant
      - Primary pattern: {prefix}_{activity}_{position}.csv
      - Fallback:        {prefix}_{activity}_2_{position}.csv (etc.)
    Returns the path if found, else None.
    """
    prefix = SENSOR_PREFIXES[sensor_key]

    # Try standard name first
    path = os.path.join(data_dir, f'{prefix}_{activity}_{position}.csv')
    if os.path.isfile(path):
        return path

    # Try _2, _3 variants
    for suffix in ['_2', '_3']:
        path = os.path.join(data_dir, f'{prefix}_{activity}{suffix}_{position}.csv')
        if os.path.isfile(path):
            return path

    return None


def read_csv(path):
    """
    Read a sensor CSV file.
    Format: id,attr_time,attr_x,attr_y,attr_z
    Returns: (timestamps: ndarray (N,), values: ndarray (N, 3)) or (None, None)
    """
    try:
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
    except Exception as e:
        print(f'      WARNING: Could not read {os.path.basename(path)}: {e}')
        return None, None

    if data.ndim != 2 or data.shape[1] < 4:
        print(f'      WARNING: Unexpected shape in {os.path.basename(path)}: {data.shape}')
        return None, None

    # Drop NaN rows
    valid = ~np.isnan(data).any(axis=1)
    data = data[valid]

    if len(data) < WINDOW_SIZE:
        print(f'      WARNING: Too few samples in {os.path.basename(path)} '
              f'({len(data)} < {WINDOW_SIZE}), skipping')
        return None, None

    # Columns: id(0), attr_time(1), attr_x(2), attr_y(3), attr_z(4)
    timestamps = data[:, 1]
    values = data[:, 2:5].astype(np.float32)

    return timestamps, values


# ─── Sensor Alignment ───────────────────────────────────────────────────────

def align_sensors(acc_ts, acc_val, gyr_ts, gyr_val, mag_ts, mag_val):
    """
    Align gyroscope and magnetometer readings to accelerometer timestamps
    using nearest-neighbour interpolation.

    Returns: ndarray (N, 9) — [acc_xyz, gyr_xyz, mag_xyz]
    """
    N = len(acc_ts)
    aligned = np.zeros((N, 9), dtype=np.float32)
    aligned[:, 0:3] = acc_val

    # Align gyroscope
    gyr_idx = np.searchsorted(gyr_ts, acc_ts)
    gyr_idx = np.clip(gyr_idx, 0, len(gyr_ts) - 1)
    # Also check the previous index to find the true nearest
    prev_idx = np.clip(gyr_idx - 1, 0, len(gyr_ts) - 1)
    use_prev = np.abs(gyr_ts[prev_idx] - acc_ts) < np.abs(gyr_ts[gyr_idx] - acc_ts)
    gyr_idx[use_prev] = prev_idx[use_prev]
    aligned[:, 3:6] = gyr_val[gyr_idx]

    # Align magnetometer
    mag_idx = np.searchsorted(mag_ts, acc_ts)
    mag_idx = np.clip(mag_idx, 0, len(mag_ts) - 1)
    prev_idx = np.clip(mag_idx - 1, 0, len(mag_ts) - 1)
    use_prev = np.abs(mag_ts[prev_idx] - acc_ts) < np.abs(mag_ts[mag_idx] - acc_ts)
    mag_idx[use_prev] = prev_idx[use_prev]
    aligned[:, 6:9] = mag_val[mag_idx]

    return aligned


# ─── Windowing ───────────────────────────────────────────────────────────────

def segment_windows(data, label, subject_id):
    """
    Segment (N, 9) data into non-overlapping windows of WINDOW_SIZE.
    Returns: windows list [(128, 9)], labels list, subject_ids list
    """
    step = WINDOW_SIZE - WINDOW_OVERLAP
    windows, labels, subjects = [], [], []

    for start in range(0, len(data) - WINDOW_SIZE + 1, step):
        window = data[start:start + WINDOW_SIZE]
        windows.append(window)
        labels.append(label)
        subjects.append(subject_id)

    return windows, labels, subjects


# ─── Main Processing Loop ───────────────────────────────────────────────────

def main():
    all_windows = []
    all_labels = []
    all_subjects = []

    for subj_id in range(1, NUM_SUBJECTS + 1):
        data_dir = os.path.join(RAW_DATA_DIR, f'proband{subj_id}', 'data')
        if not os.path.isdir(data_dir):
            print(f'[Subject {subj_id:2d}] MISSING — skipping')
            continue

        print(f'[Subject {subj_id:2d}] Processing...')
        subj_total = 0

        for activity, label in sorted(ACTIVITIES.items(), key=lambda x: x[1]):
            # Find all 3 sensor CSVs
            sensor_data = {}
            skip = False
            for sensor_key in ['acc', 'gyr', 'mag']:
                csv_path = find_csv(data_dir, sensor_key, activity, BODY_POSITION)
                if csv_path is None:
                    print(f'    {activity:20s}: SKIPPED (missing {sensor_key} CSV)')
                    skip = True
                    break
                ts, vals = read_csv(csv_path)
                if ts is None:
                    skip = True
                    break
                sensor_data[sensor_key] = (ts, vals)

            if skip:
                continue

            # Align sensors
            aligned = align_sensors(
                sensor_data['acc'][0], sensor_data['acc'][1],
                sensor_data['gyr'][0], sensor_data['gyr'][1],
                sensor_data['mag'][0], sensor_data['mag'][1],
            )

            # Segment into windows
            windows, labels, subjects = segment_windows(aligned, label, subj_id)
            n_samples = len(aligned)

            print(f'    {activity:20s}: {len(windows):4d} windows '
                  f'({n_samples} samples, {n_samples / SAMPLE_RATE:.1f}s)')

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_subjects.extend(subjects)
            subj_total += len(windows)

        print(f'  => {subj_total} total windows\n')

    # ── Stack arrays ─────────────────────────────────────────────────────
    X = np.array(all_windows, dtype=np.float32)          # (N, 128, 9)
    y_int = np.array(all_labels, dtype=np.int32)         # (N,)
    subjects = np.array(all_subjects, dtype=np.int32)    # (N,)

    # ── Scaling (StandardScaler per channel) ─────────────────────────────
    print('Scaling data...')
    N = X.shape[0]
    X_flat = X.reshape(-1, 9)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(N, WINDOW_SIZE, 9).astype(np.float32)

    # ── One-hot encode ───────────────────────────────────────────────────
    y_onehot = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    y_onehot[np.arange(N), y_int] = 1.0

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'X_realworld_loso.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_realworld_loso.npy'), y_onehot)
    np.save(os.path.join(OUTPUT_DIR, 'subjects_realworld_loso.npy'), subjects)

    # ── Summary ──────────────────────────────────────────────────────────
    print('=' * 60)
    print('PREPROCESSING COMPLETE')
    print(f'  Total windows:  {N}')
    print(f'  X shape:        {X.shape}')
    print(f'  y shape:        {y_onehot.shape}')
    print(f'  Subjects:       {np.unique(subjects)}')
    print()
    print('  Samples per subject:')
    for s in np.unique(subjects):
        print(f'    Subject {s:2d}: {(subjects == s).sum():5d} windows')
    print()
    print('  Samples per activity:')
    activity_names = {v: k for k, v in ACTIVITIES.items()}
    for i in range(NUM_CLASSES):
        name = activity_names[i]
        count = (y_int == i).sum()
        print(f'    {name:20s}: {count:5d} windows')
    print('=' * 60)


if __name__ == '__main__':
    main()
