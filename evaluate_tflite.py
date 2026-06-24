import numpy as np
import tensorflow as tf
import tensorflow as tf

def test_tflite_model(tflite_model_path, X_test, y_test):
    """Test a TFLite model against the test dataset"""
    print(f"Loading TFLite model from '{tflite_model_path}'...")
    
    # Initialize TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    # Determine if input needs to be quantized (int8)
    input_dtype = input_details[0]['dtype']
    is_quantized = input_dtype == np.int8
    
    if is_quantized:
        print("Model is INT8 Quantized. Applying input scaling...")
        scale, zero_point = input_details[0]['quantization']
        # Quantize the test data
        X_test_quant = np.round(X_test / scale + zero_point).astype(np.int8)
    else:
        print("Model is Float32.")
        X_test_quant = X_test.astype(np.float32)

    print("Running inference...")
    y_pred = []
    
    # Run inference for every sample
    for i in range(len(X_test_quant)):
        # Expand dims to match batch size 1
        input_data = np.expand_dims(X_test_quant[i], axis=0)
        
        # Set tensor and invoke
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
        # Get result
        output_data = interpreter.get_tensor(output_index)
        y_pred.append(output_data[0])
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(X_test_quant)} samples...")
            
    y_pred = np.array(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate Accuracy
    correct = np.sum(y_pred_classes == y_true_classes)
    accuracy = correct / len(y_test)
    
    print("\n" + "="*50)
    print(f"Results for {tflite_model_path}")
    print("="*50)
    print(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{len(y_test)})")
    
    print("\nClassification Report:")
    ACTIVITY_LABELS = ['Climbing Down', 'Climbing Up', 'Jumping', 'Lying', 'Running', 'Sitting', 'Standing', 'Walking']
    
    # Calculate per-class metrics natively
    for i, label in enumerate(ACTIVITY_LABELS):
        true_pos = np.sum((y_pred_classes == i) & (y_true_classes == i))
        actual = np.sum(y_true_classes == i)
        predicted = np.sum(y_pred_classes == i)
        
        recall = true_pos / actual if actual > 0 else 0
        precision = true_pos / predicted if predicted > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label:15s} Support: {actual:4d} | Precision: {precision:4.2f} | Recall: {recall:4.2f} | F1: {f1:4.2f}")
    
    return accuracy
    
def main():
    print("Loading test dataset...")
    X = np.load('assets/X_realworldhar_raw_scaled.npy')
    y = np.load('assets/y_realworldhar_one_hot.npy')
    
    # Use the identical stratified split from training to get the exact same test set
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    y_classes = np.argmax(y, axis=1)
    np.random.seed(42)
    
    for c in range(y.shape[1]):
        indices = np.where(y_classes == c)[0]
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.8) # 80% train, 20% test
        
        test_idx = indices[split_idx:]
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])
        
    X_test = np.vstack(X_test_list)
    y_test = np.vstack(y_test_list)
    
    # Needs to match the shuffle seed from the original script
    test_shuffle = np.random.permutation(len(X_test))
    X_test, y_test = X_test[test_shuffle], y_test[test_shuffle]
    
    # Run evaluations
    print("\n--- Evaluating INT8 Model ---")
    int8_acc = test_tflite_model('models/har_realworld_cnn_tinyml_int8.tflite', X_test, y_test)
    
    print("\n--- Evaluating Float32 Model ---")
    float_acc = test_tflite_model('models/har_realworld_cnn_tinyml.tflite', X_test, y_test)
    
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Float32 Accuracy: {float_acc*100:.2f}%")
    print(f"INT8 Accuracy:    {int8_acc*100:.2f}%")
    print(f"Accuracy Drop:    {(float_acc - int8_acc)*100:.2f}%")
    
if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings
    main()
