#!/usr/bin/env python3
"""
Lightweight 1D CNN for Human Activity Recognition
Subject-Aware (LOSO) Cross Validation on RealWorld HAR Dataset

This script uses the IDENTICAL data loading and LOSO partitioning as
train_realworld_cnn_gru_loso.py, but with a simpler CNN-only architecture
(no GRU, no attention, no feature augmentation) for comparison purposes.

Architecture:
  Conv1D(32, k=5) → BN → ReLU → MaxPool
  Conv1D(64, k=5) → BN → ReLU → MaxPool
  GlobalAveragePooling1D
  Dense(32) → Dropout → Softmax(8)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D,
    Dense, GlobalAveragePooling1D, ReLU
)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
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


# ═══════════════════════════════════════════════════════════
# Model Architecture — Simple 2-Block CNN
# ═══════════════════════════════════════════════════════════

def build_model(input_shape, num_classes):
    """
    Lightweight 2-block 1D CNN for HAR.

    Architecture:
      Conv1D(32, k=5) → BN → ReLU → MaxPool → Dropout(0.2)
      Conv1D(64, k=5) → BN → ReLU → MaxPool → Dropout(0.2)
      GlobalAveragePooling1D
      Dense(32) → Dropout(0.3) → Softmax

    This is intentionally simpler than the CNN-GRU-Attention model
    to serve as a baseline comparison.
    """
    inputs = Input(shape=input_shape, name='input')

    # ── Conv Block 1 ──────────────────────────────────────
    x = Conv1D(32, kernel_size=5, padding='same', name='conv1d_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = ReLU(name='relu_1')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)

    # ── Conv Block 2 ──────────────────────────────────────
    x = Conv1D(64, kernel_size=5, padding='same', name='conv1d_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = ReLU(name='relu_2')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)

    # ── Global Pooling ────────────────────────────────────
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)

    # ── Classification Head ───────────────────────────────
    x = Dense(32, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_dense')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_LOSO_RealWorld')
    return model


# ═══════════════════════════════════════════════════════════
# Data Loading (identical to CNN-GRU LOSO script)
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
# Stratified Subject Fold Generation (identical to CNN-GRU)
# ═══════════════════════════════════════════════════════════

def _dominant_activity(subject_id, subjects, y_labels):
    """Return the most common class index for a given subject."""
    mask = subjects == subject_id
    return int(np.argmax(np.bincount(np.argmax(y_labels[mask], axis=1))))


def generate_stratified_folds(unique_subjects, subjects, y_labels, n_folds=5):
    """
    Stratified subject grouping: subjects ranked by dominant activity,
    distributed round-robin across folds.

    IDENTICAL to the CNN-GRU LOSO script to ensure fair comparison.
    """
    activity_buckets = defaultdict(list)
    for s in unique_subjects:
        dom = _dominant_activity(s, subjects, y_labels)
        activity_buckets[dom].append(s)

    fold_buckets = [[] for _ in range(n_folds)]
    for activity_list in activity_buckets.values():
        np.random.shuffle(activity_list)
        for idx, subj in enumerate(activity_list):
            fold_buckets[idx % n_folds].append(subj)

    folds = []
    for i in range(n_folds):
        test_subjects = fold_buckets[i]
        remaining = [s for s in unique_subjects if s not in test_subjects]
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
    """Compute inverse-frequency class weights from one-hot training labels."""
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
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Oranges', vmin=0, vmax=1)
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
                   color='darkorange', alpha=0.8, edgecolor='white')
    ax.axhline(np.mean(fold_accuracies) * 100, color='tomato',
               linestyle='--', linewidth=1.5, label=f'Mean {np.mean(fold_accuracies)*100:.1f}%')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Per-fold Test Accuracy — CNN LOSO')
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
    print("Lightweight CNN — RealWorld HAR")
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

    # NO feature augmentation — use raw 9-channel sensor data
    input_shape = (X.shape[1], X.shape[2])
    print(f"Input shape for model: {input_shape} (raw, no feature augmentation)")

    # ── 2. Stratified Fold Generation ────────────────────
    folds = generate_stratified_folds(unique_subjects, subjects, y, n_folds=5)

    # ── 3. Training Config ───────────────────────────────
    output_dir = 'models/loso_cnn'
    os.makedirs(output_dir, exist_ok=True)
    fold_accuracies  = []
    aggregate_y_true = []
    aggregate_y_pred = []
    base_epochs  = 80
    batch_size   = 128

    # ── 4. Fold Loop ─────────────────────────────────────
    for fold_idx, fold in enumerate(folds, start=1):
        print("\n" + "=" * 65)
        print(f"FOLD {fold_idx}/5")
        print(f"  Train subjects : {fold['train']}")
        print(f"  Val   subjects : {fold['val']}")
        print(f"  Test  subjects : {fold['test']}")
        print("=" * 65)

        X_train, y_train, X_val, y_val, X_test, y_test = \
            prepare_fold_data(X, y, subjects, fold)

        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

        # Class weights
        class_weights = compute_fold_class_weights(y_train)
        print(f"  Class weights: { {ACTIVITY_LABELS[k]: round(v,2) for k,v in class_weights.items()} }")

        # Build model
        model = build_model(input_shape, n_classes)
        if fold_idx == 1:
            model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model_path = f'{output_dir}/har_realworld_cnn_fold{fold_idx}.keras'
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
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

    # ── 5. Aggregate Results ─────────────────────────────
    print("\n" + "=" * 65)
    print("CNN LOSO CROSS-VALIDATION SUMMARY")
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

    # ── 6. Plots & Report ────────────────────────────────
    plot_confusion_matrix(
        cm, class_names,
        title=f'CNN LOSO Aggregate CM\n'
              f'Accuracy: {avg_acc * 100:.2f}% (±{std_acc * 100:.2f}%)',
        filename=f'{output_dir}/aggregate_confusion_matrix_cnn.png'
    )
    plot_fold_accuracies(
        fold_accuracies,
        filename=f'{output_dir}/fold_accuracies_cnn.png'
    )

    report_path = f'{output_dir}/results_cnn_loso.txt'
    with open(report_path, 'w') as f:
        f.write("CNN LOSO CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generalization Accuracy: {avg_acc*100:.2f}% (±{std_acc*100:.2f}%)\n\n")
        f.write("Architecture:\n")
        f.write("  Conv1D(32, k=5) → BN → ReLU → MaxPool\n")
        f.write("  Conv1D(64, k=5) → BN → ReLU → MaxPool\n")
        f.write("  GlobalAvgPool → Dense(32) → Dropout → Softmax(8)\n\n")
        f.write("No feature augmentation (raw 9-channel sensor data)\n\n")
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
