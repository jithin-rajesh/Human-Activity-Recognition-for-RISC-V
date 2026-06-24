#!/usr/bin/env python3
"""
Improved Hybrid CNN-GRU with Attention for Human Activity Recognition
Optimised for RealWorld HAR Dataset with Subject-Aware Cross Validation
(Leave-Group-Out / LOSO Evaluation)

Key improvements over baseline:
1. Multi-scale CNN (parallel convolutions at kernel sizes 3, 7, 11)
2. Two-layer stacked GRU (64 units) for longer temporal context
3. Gravity-separated features appended (low-pass filtered acceleration)
4. Signal Magnitude Area (SMA) + tilt angle as engineered features
5. Class-weighted loss to address Sitting/Standing underperformance
6. Stratified subject grouping in fold generation
7. Larger attention (64 units) matching increased GRU capacity
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D,
    Bidirectional, GRU, Dense, Add, GaussianNoise, Concatenate,
    GlobalAveragePooling1D, LayerNormalization
)
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, filtfilt
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ─────────────────────────────────────────────
# Activity labels for RealWorld HAR (8 classes)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Sensor Config
# The RealWorld HAR dataset provides raw triaxial accelerometer data.
# We assume axes 0,1,2 of each sensor are X,Y,Z acceleration (m/s² or g).
# If multiple sensor positions are stacked in X, set ACCEL_AXIS_GROUPS
# to list the (start, end) index slices of each sensor's XYZ axes.
# E.g. for 3 sensors × 3 axes = 9 features:
#   ACCEL_AXIS_GROUPS = [(0,3), (3,6), (6,9)]
# Set to None to auto-detect (assumes first 3 columns are one accelerometer).
# ─────────────────────────────────────────────
ACCEL_AXIS_GROUPS = None  # auto-detect: use first 3 axes only
GRAVITY_CUTOFF_HZ = 0.3   # low-pass cutoff for gravity separation
SAMPLING_RATE_HZ  = 50    # RealWorld HAR default sampling rate


# ═══════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════

def _butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def extract_gravity_features(X_windows, accel_groups=None, cutoff=GRAVITY_CUTOFF_HZ, fs=SAMPLING_RATE_HZ):
    """
    For each accelerometer group in a window, separate gravity (low-pass)
    from dynamic acceleration (residual).  Returns:
        gravity components  (shape: N × T × n_gravity_axes)
        dynamic components  (shape: N × T × n_gravity_axes)
    appended as new feature channels.

    Parameters
    ----------
    X_windows : ndarray (N, T, F)
    accel_groups : list of (start, end) tuples, or None
        If None, uses the first 3 feature axes as one accelerometer group.
    """
    n_windows, T, F = X_windows.shape

    if accel_groups is None:
        accel_groups = [(0, min(3, F))]

    b, a = _butter_lowpass(cutoff, fs)

    gravity_channels = []
    dynamic_channels = []

    for (start, end) in accel_groups:
        segment = X_windows[:, :, start:end]          # (N, T, 3)
        gravity = np.zeros_like(segment)
        for n in range(n_windows):
            for ax in range(segment.shape[2]):
                gravity[n, :, ax] = filtfilt(b, a, segment[n, :, ax])
        dynamic = segment - gravity
        gravity_channels.append(gravity)
        dynamic_channels.append(dynamic)

    gravity_arr = np.concatenate(gravity_channels, axis=2)   # (N, T, n_grav)
    dynamic_arr = np.concatenate(dynamic_channels, axis=2)   # (N, T, n_dyn)
    return gravity_arr, dynamic_arr


def extract_scalar_features(X_windows, accel_groups=None):
    """
    Compute per-timestep scalar features:
      - Signal Magnitude Area proxy: mean(|x| + |y| + |z|) per timestep
      - Tilt angle estimate: arctan2(gravity_z, sqrt(gx² + gy²))

    Returns array of shape (N, T, n_scalar_features).
    """
    if accel_groups is None:
        accel_groups = [(0, min(3, X_windows.shape[2]))]

    all_scalar = []

    for (start, end) in accel_groups:
        seg = X_windows[:, :, start:end]               # (N, T, 3)

        # SMA-like: L1 norm per timestep
        sma = np.sum(np.abs(seg), axis=2, keepdims=True)   # (N, T, 1)

        # Tilt angle (requires gravity separation)
        b, a = _butter_lowpass(GRAVITY_CUTOFF_HZ, SAMPLING_RATE_HZ)
        gravity = np.zeros_like(seg)
        for n in range(seg.shape[0]):
            for ax in range(seg.shape[2]):
                gravity[n, :, ax] = filtfilt(b, a, seg[n, :, ax])

        gx = gravity[:, :, 0]
        gy = gravity[:, :, 1]
        gz = gravity[:, :, 2] if seg.shape[2] > 2 else np.zeros_like(gx)
        tilt = np.arctan2(gz, np.sqrt(gx**2 + gy**2 + 1e-9))[:, :, np.newaxis]   # (N,T,1)

        all_scalar.extend([sma, tilt])

    return np.concatenate(all_scalar, axis=2).astype(np.float32)


def augment_features(X_windows, accel_groups=None):
    """
    Augment raw windows with gravity, dynamic, and scalar channels.
    Returns X_augmented of shape (N, T, F_augmented).
    """
    if accel_groups is None:
        accel_groups = [(0, min(3, X_windows.shape[2]))]

    gravity_arr, dynamic_arr = extract_gravity_features(X_windows, accel_groups)
    scalar_arr = extract_scalar_features(X_windows, accel_groups)

    X_aug = np.concatenate([X_windows, gravity_arr, dynamic_arr, scalar_arr], axis=2)
    print(f"Feature augmentation: {X_windows.shape[2]} → {X_aug.shape[2]} channels")
    return X_aug.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# Attention Layer
# ═══════════════════════════════════════════════════════════

class SelfAttention(layers.Layer):
    """Additive self-attention (Bahdanau-style) over time axis."""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W = self.add_weight(name='W', shape=(dim, self.units),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,),
                                  initializer='zeros', trainable=True)
        self.u = self.add_weight(name='u', shape=(self.units,),
                                  initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[-1], [0]]) + self.b)
        weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[[-1], [0]]), axis=1
        )                                                          # (N, T)
        weights = tf.expand_dims(weights, -1)                      # (N, T, 1)
        context = tf.reduce_sum(x * weights, axis=1)               # (N, D)
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg


# ═══════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════

def build_model(input_shape, num_classes):
    """
    Multi-scale CNN → Stacked Bidirectional GRU → Self-Attention → Dense

    Improvements vs baseline:
    - Parallel CNN branches at kernel sizes 3, 7, 11 (multi-scale temporal features)
    - Two stacked Bidirectional GRU layers (64 units each)
    - Attention units increased to 64
    - LayerNorm before GRU for training stability
    """
    inputs = Input(shape=input_shape, name='input')
    x = GaussianNoise(0.1, name='gaussian_noise')(inputs)  # Increased noise to 0.1

    # ── Multi-Scale CNN Block ──────────────────────────────
    def cnn_branch(inp, kernel_size, filters, name_prefix):
        c = Conv1D(filters, kernel_size=kernel_size, padding='same',
                   activation='relu', name=f'{name_prefix}_conv',
                   kernel_regularizer=l2(1e-4))(inp)
        c = BatchNormalization(name=f'{name_prefix}_bn')(c)
        return c

    branch3  = cnn_branch(x, kernel_size=3,  filters=32, name_prefix='k3')
    branch7  = cnn_branch(x, kernel_size=7,  filters=32, name_prefix='k7')
    branch11 = cnn_branch(x, kernel_size=11, filters=32, name_prefix='k11')

    # Concatenate multi-scale features → 96 channels
    merged = Concatenate(name='multiscale_concat')([branch3, branch7, branch11])

    # Shared refinement conv
    refined = Conv1D(64, kernel_size=3, padding='same', activation='relu',
                     kernel_regularizer=l2(1e-4), name='refine_conv')(merged)
    refined = BatchNormalization(name='refine_bn')(refined)
    refined = Dropout(0.3, name='dropout_cnn')(refined)  # Increased from 0.2

    # Residual projection of input to 64 channels
    residual = Conv1D(64, kernel_size=1, padding='same', name='res_proj')(x)
    cnn_out  = Add(name='res_add')([refined, residual])
    cnn_out  = MaxPooling1D(pool_size=2, name='maxpool')(cnn_out)

    # ── Stacked Bidirectional GRU ──────────────────────────
    cnn_out = LayerNormalization(name='ln_pre_gru')(cnn_out)

    # Layer 1 — returns sequences for layer 2
    gru1 = Bidirectional(
        GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
            kernel_regularizer=l2(1e-4), name='gru1'),
        name='bi_gru1'
    )(cnn_out)
    gru1 = Dropout(0.3, name='dropout_gru1')(gru1)  # Increased from 0.2

    # Layer 2 — returns sequences for attention
    gru2 = Bidirectional(
        GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
            kernel_regularizer=l2(1e-4), name='gru2'),
        name='bi_gru2'
    )(gru1)
    gru2 = Dropout(0.4, name='dropout_gru2')(gru2)  # Increased from 0.3

    # ── Self-Attention ─────────────────────────────────────
    attended = SelfAttention(units=64, name='self_attention')(gru2)

    # ── Classification Head ────────────────────────────────
    dense = Dense(64, activation='relu', name='dense_1',
                  kernel_regularizer=l2(1e-4))(attended)
    dense = Dropout(0.4, name='dropout_dense')(dense)  # Increased from 0.3
    outputs = Dense(num_classes, activation='softmax', name='output')(dense)

    model = Model(inputs=inputs, outputs=outputs,
                  name='Improved_CNN_GRU_Attention_RealWorld')
    return model


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_loso_data():
    """Load preprocessed RealWorld HAR dataset with subject annotations."""
    print("Loading LOSO RealWorld HAR data...")
    X        = np.load('assets/X_realworld_loso.npy')
    y        = np.load('assets/y_realworld_loso.npy')
    subjects = np.load('assets/subjects_realworld_loso.npy')

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    print(f"Loaded {len(X)} windows, shape {X.shape}")
    unique_subjects = np.unique(subjects)
    print(f"Found {len(unique_subjects)} subjects: {unique_subjects}")
    return X, y, subjects


# ═══════════════════════════════════════════════════════════
# Stratified Subject Fold Generation
# ═══════════════════════════════════════════════════════════

def _dominant_activity(subject_id, subjects, y_labels):
    """Return the most common class index for a given subject."""
    mask = subjects == subject_id
    return int(np.argmax(np.bincount(np.argmax(y_labels[mask], axis=1))))


def generate_stratified_folds(unique_subjects, subjects, y_labels, n_folds=5):
    """
    Stratified subject grouping: subjects are ranked by dominant activity
    and distributed round-robin across folds so each fold covers a similar
    activity distribution — reduces the ±8.63% variance seen in the baseline.

    For 15 subjects / 5 folds: 3 test subjects, 2 val subjects, 10 train.
    """
    # Group subjects by dominant activity
    activity_buckets = defaultdict(list)
    for s in unique_subjects:
        dom = _dominant_activity(s, subjects, y_labels)
        activity_buckets[dom].append(s)

    # Round-robin assignment into n_folds buckets
    fold_buckets = [[] for _ in range(n_folds)]
    for activity_list in activity_buckets.values():
        np.random.shuffle(activity_list)
        for idx, subj in enumerate(activity_list):
            fold_buckets[idx % n_folds].append(subj)

    folds = []
    for i in range(n_folds):
        test_subjects = fold_buckets[i]

        remaining = [s for s in unique_subjects if s not in test_subjects]
        # pick 2 val subjects from the fold immediately after (wraps around)
        val_pool = fold_buckets[(i + 1) % n_folds]
        val_subjects  = val_pool[:2]
        train_subjects = [s for s in remaining if s not in val_subjects]

        folds.append({
            'train': np.array(train_subjects),
            'val':   np.array(val_subjects),
            'test':  np.array(test_subjects)
        })

    return folds


# ═══════════════════════════════════════════════════════════
# Fold Data Preparation
# ═══════════════════════════════════════════════════════════

def prepare_fold_data(X, y, subjects, fold_config):
    """Filter dataset by subject allocation and shuffle training set."""
    train_mask = np.isin(subjects, fold_config['train'])
    val_mask   = np.isin(subjects, fold_config['val'])
    test_mask  = np.isin(subjects, fold_config['test'])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ═══════════════════════════════════════════════════════════
# Class Weights
# ═══════════════════════════════════════════════════════════

def compute_fold_class_weights(y_train_ohe):
    """
    Compute inverse-frequency class weights from one-hot training labels.
    Upweights minority/hard classes (Sitting, Standing) automatically.
    """
    y_int = np.argmax(y_train_ohe, axis=1)
    classes = np.unique(y_int)
    weights = compute_class_weight(class_weight='balanced',
                                   classes=classes, y=y_int)
    return dict(zip(classes, weights))


# ═══════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, title='Aggregate Confusion Matrix',
                          filename='cm.png'):
    if not MATPLOTLIB_AVAILABLE:
        return
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='Row-normalised Accuracy')

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks);  ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(ticks);  ax.set_yticklabels(class_names)

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i,
                    f"{cm[i,j]}\n({cm_pct[i,j]*100:.1f}%)",
                    ha='center', va='center', fontsize=7,
                    color='white' if cm_pct[i, j] > thresh else 'black')

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved → {filename}")


def plot_fold_accuracies(fold_accuracies, filename='fold_accuracies.png'):
    if not MATPLOTLIB_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = np.arange(1, len(fold_accuracies) + 1)
    bars  = ax.bar(folds, [a * 100 for a in fold_accuracies],
                   color='steelblue', alpha=0.8, edgecolor='white')
    ax.axhline(np.mean(fold_accuracies) * 100, color='tomato',
               linestyle='--', linewidth=1.5, label=f'Mean {np.mean(fold_accuracies)*100:.1f}%')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Per-fold Test Accuracy (LOSO)')
    ax.legend()
    for bar, acc in zip(bars, fold_accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fold accuracy plot saved → {filename}")


# ═══════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Improved Hybrid CNN-GRU-Attention — RealWorld HAR")
    print("Subject-Aware (LOSO) Cross Validation")
    print("=" * 65)

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")

    np.random.seed(42)
    tf.random.set_seed(42)

    # ── 1. Load Raw Data ─────────────────────────────────
    X, y, subjects = load_loso_data()
    unique_subjects = np.unique(subjects)
    n_classes = y.shape[1]

    # ── 2. Feature Augmentation ──────────────────────────
    # Detect accelerometer axis groups from feature count
    # RealWorld HAR: commonly 3 or 9 axes depending on preprocessing.
    n_features = X.shape[2]
    if n_features >= 9:
        # 3 sensors × 3 axes  (e.g. thigh, chest, forearm)
        accel_groups = [(0, 3), (3, 6), (6, 9)]
    elif n_features >= 6:
        accel_groups = [(0, 3), (3, 6)]
    else:
        accel_groups = [(0, min(3, n_features))]

    print(f"\nDetected accelerometer groups: {accel_groups}")
    X_aug = augment_features(X, accel_groups=accel_groups)
    input_shape = (X_aug.shape[1], X_aug.shape[2])
    print(f"Input shape for model: {input_shape}")

    # ── 3. Stratified Fold Generation ────────────────────
    folds = generate_stratified_folds(unique_subjects, subjects, y, n_folds=5)

    # ── 4. Training Config ───────────────────────────────
    os.makedirs('models/loso', exist_ok=True)
    fold_accuracies  = []
    aggregate_y_true = []
    aggregate_y_pred = []
    base_epochs  = 80
    batch_size   = 128

    # ── 5. Fold Loop ─────────────────────────────────────
    for fold_idx, fold in enumerate(folds, start=1):
        print("\n" + "=" * 65)
        print(f"FOLD {fold_idx}/5")
        print(f"  Train subjects : {fold['train']}")
        print(f"  Val   subjects : {fold['val']}")
        print(f"  Test  subjects : {fold['test']}")
        print("=" * 65)

        X_train, y_train, X_val, y_val, X_test, y_test = \
            prepare_fold_data(X_aug, y, subjects, fold)

        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

        # Class weights (upweight Sitting, Standing automatically)
        class_weights = compute_fold_class_weights(y_train)
        print(f"  Class weights: { {ACTIVITY_LABELS[k]: round(v,2) for k,v in class_weights.items()} }")

        # Build model
        model = build_model(input_shape, n_classes)
        if fold_idx == 1:
            model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Reduced from 0.001
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model_path = f'models/loso/har_realworld_improved_fold{fold_idx}.keras'
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=12,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4,
                min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_accuracy',
                save_best_only=True, verbose=0
            )
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=base_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFold {fold_idx} Test Accuracy: {accuracy * 100:.2f}%")
        fold_accuracies.append(accuracy)

        y_pred = model.predict(X_test, verbose=0)
        aggregate_y_pred.extend(np.argmax(y_pred, axis=1))
        aggregate_y_true.extend(np.argmax(y_test, axis=1))

        keras.backend.clear_session()

    # ── 6. Aggregate Results ─────────────────────────────
    print("\n" + "=" * 65)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1}: {acc * 100:.2f}%")

    avg_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nGeneralization Accuracy: {avg_acc * 100:.2f}% (±{std_acc * 100:.2f}%)")

    class_names = [ACTIVITY_LABELS[i] for i in range(n_classes)]
    cm = confusion_matrix(aggregate_y_true, aggregate_y_pred)

    print("\nAggregate Classification Report:")
    print(classification_report(aggregate_y_true, aggregate_y_pred,
                                 target_names=class_names))

    # ── 7. Plots & Report ────────────────────────────────
    plot_confusion_matrix(
        cm, class_names,
        title=f'Improved LOSO Aggregate CM\n'
              f'Accuracy: {avg_acc * 100:.2f}% (±{std_acc * 100:.2f}%)',
        filename='models/loso/aggregate_confusion_matrix_improved.png'
    )
    plot_fold_accuracies(
        fold_accuracies,
        filename='models/loso/fold_accuracies_improved.png'
    )

    report_path = 'models/loso/results_improved.txt'
    with open(report_path, 'w') as f:
        f.write("IMPROVED LOSO CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generalization Accuracy: {avg_acc*100:.2f}% (±{std_acc*100:.2f}%)\n\n")
        f.write("Improvements applied:\n")
        f.write("  1. Multi-scale CNN (kernels 3, 7, 11)\n")
        f.write("  2. Stacked Bidirectional GRU (2× 64 units)\n")
        f.write("  3. Gravity-separated features (low-pass @ 0.3 Hz)\n")
        f.write("  4. SMA + tilt angle engineered features\n")
        f.write("  5. Class-weighted loss (balanced)\n")
        f.write("  6. Stratified subject fold generation\n")
        f.write("  7. LayerNorm before GRU stack\n\n")
        f.write("Per-fold Accuracies:\n")
        for i, acc in enumerate(fold_accuracies):
            f.write(f"  Fold {i+1}: {acc*100:.2f}%\n")
        f.write("\nAggregate Classification Report:\n")
        f.write(classification_report(aggregate_y_true, aggregate_y_pred,
                                       target_names=class_names))

    print(f"\nResults saved → {report_path}")
    print("Done.")


if __name__ == '__main__':
    main()