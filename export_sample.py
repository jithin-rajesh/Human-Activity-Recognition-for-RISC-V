import numpy as np
import tensorflow as tf
import sys

def main(num_samples=50):
    # Load TFLite model to get quantization parameters
    interpreter = tf.lite.Interpreter(model_path='models/har_cnn_tinyml_int8.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    
    # Load dataset (UCI HAR)
    X_test = np.load('assets/X_test_raw_scaled.npy')
    y_test = np.load('assets/y_test_one_hot.npy')
        
    # Get indices and shuffle them for a diverse batch
    indices = np.arange(len(X_test))
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(indices)
    
    samples = X_test[indices[:num_samples]]
    labels = np.argmax(y_test[indices[:num_samples]], axis=1)
    
    scale, zero_point = input_details['quantization']
    
    # Quantize inputs to int8
    # Formula: quantized_value = (value / scale) + zero_point
    samples_quantized = np.round(samples / scale + zero_point).astype(np.int8)
    
    # Export as C header
    print("// Auto-generated test data - " + str(num_samples) + " shuffled samples (UCI HAR)")
    print("#include <stdint.h>")
    print("const int kNumSamples = " + str(num_samples) + ";")
    print("const int kSampleDataSize = " + str(samples.shape[1] * samples.shape[2]) + ";")
    
    vals = samples_quantized.flatten().tolist()
    print("const int8_t batch_samples[] = { " + ", ".join(map(str, vals)) + " };")
    
    label_vals = labels.tolist()
    print("const int batch_labels[] = { " + ", ".join(map(str, label_vals)) + " };")

if __name__ == "__main__":
    main()
