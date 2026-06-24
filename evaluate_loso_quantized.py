#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict

# Activity labels
ACTIVITY_LABELS = {
    0: 'Climbing Down', 1: 'Climbing Up', 2: 'Jumping', 3: 'Lying',
    4: 'Running', 5: 'Sitting', 6: 'Standing', 7: 'Walking'
}

def load_loso_data():
    X        = np.load('assets/X_realworld_loso.npy')
    y        = np.load('assets/y_realworld_loso.npy')
    subjects = np.load('assets/subjects_realworld_loso.npy')
    return X.astype(np.float32), y.astype(np.float32), subjects

def _dominant_activity(subject_id, subjects, y_labels):
    mask = subjects == subject_id
    return int(np.argmax(np.bincount(np.argmax(y_labels[mask], axis=1))))

def generate_stratified_folds(unique_subjects, subjects, y_labels, n_folds=5):
    activity_buckets = defaultdict(list)
    for s in unique_subjects:
        dom = _dominant_activity(s, subjects, y_labels)
        activity_buckets[dom].append(s)

    fold_buckets = [[] for _ in range(n_folds)]
    
    # Needs to match EXACT seed from training
    np.random.seed(42)  
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

def prepare_fold_data(X, y, subjects, fold_config):
    train_mask = np.isin(subjects, fold_config['train'])
    test_mask  = np.isin(subjects, fold_config['test'])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    np.random.seed(42)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    return X_train, y_train, X_test, y_test

def evaluate_tflite_model(interpreter, X_test, y_test):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    is_quantized = input_details[0]['dtype'] == np.int8
    
    if is_quantized:
        scale, zero_point = input_details[0]['quantization']
        X_test_quant = np.round(X_test / scale + zero_point).astype(np.int8)
    else:
        X_test_quant = X_test.astype(np.float32)

    y_pred = []
    for i in range(len(X_test_quant)):
        input_data = np.expand_dims(X_test_quant[i], axis=0)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        y_pred.append(output_data[0])
        
    y_pred = np.array(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    correct = np.sum(y_pred_classes == y_true_classes)
    accuracy = correct / len(y_test)
    return accuracy

def main():
    X, y, subjects = load_loso_data()
    unique_subjects = np.unique(subjects)
    folds = generate_stratified_folds(unique_subjects, subjects, y, n_folds=5)

    base_dir = 'models/loso_cnn'
    
    fp32_accuracies = []
    int8_accuracies = []

    print("="*60)
    print("Evaluating Quantized CNN using LOSO")
    print("="*60)

    for i, fold in enumerate(folds, 1):
        keras_path = f"{base_dir}/har_realworld_cnn_fold{i}.keras"
        int8_path = f"{base_dir}/har_realworld_cnn_fold{i}_int8.tflite"
        float32_path = f"{base_dir}/har_realworld_cnn_fold{i}_float32.tflite"
        
        if not os.path.exists(keras_path):
            print(f"Skipping Fold {i}: Model not found at {keras_path}")
            continue
            
        print(f"\nProcessing Fold {i} / 5")
        X_train, y_train, X_test, y_test = prepare_fold_data(X, y, subjects, fold)

        # Load Keras model to evaluate precise FP32 baseline
        model = tf.keras.models.load_model(keras_path)
        loss, val_acc_keras = model.evaluate(X_test, y_test, verbose=0)
        fp32_accuracies.append(val_acc_keras)

        # Convert to INT8 TFLite
        print("  Converting to INT8 TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Use 500 samples from THIS Fold's specific train set for calibration
        def representative_dataset():
            indices = np.random.permutation(len(X_train))[:500]
            for idx in indices:
                yield [X_train[idx:idx+1]]
                
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_int8_model = converter.convert()
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8_model)
            
        # Parse INT8 TFLite
        interpreter = tf.lite.Interpreter(model_content=tflite_int8_model)
        interpreter.allocate_tensors()
        
        # Evaluate INT8
        int8_acc = evaluate_tflite_model(interpreter, X_test, y_test)
        int8_accuracies.append(int8_acc)
        
        print(f"  Fold {i} FP32 Accuracy: {val_acc_keras*100:.2f}%")
        print(f"  Fold {i} INT8 Accuracy: {int8_acc*100:.2f}%")
        print(f"  Drop: {(val_acc_keras - int8_acc)*100:.2f}%")

    print("\n" + "="*60)
    print("LOSO CROSS-VALIDATION SUMMARY (QUANTIZED)")
    print("="*60)
    
    mean_fp32 = np.mean(fp32_accuracies)
    std_fp32  = np.std(fp32_accuracies)
    mean_int8 = np.mean(int8_accuracies)
    std_int8  = np.std(int8_accuracies)
    
    print(f"Generalization Accuracy (FP32 Keras): {mean_fp32 * 100:.2f}% (±{std_fp32 * 100:.2f}%)")
    print(f"Generalization Accuracy (INT8 TFLite): {mean_int8 * 100:.2f}% (±{std_int8 * 100:.2f}%)")
    print(f"Average Accuracy Drop from Quantization: {(mean_fp32 - mean_int8) * 100:.2f}%")

if __name__ == '__main__':
    # Ignore deprecation warnings for cleaner output
    import os, warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    main()
