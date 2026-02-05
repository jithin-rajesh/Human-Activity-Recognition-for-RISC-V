
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Activity labels for UCI HAR dataset
ACTIVITY_LABELS = {
    0: 'Walking',
    1: 'Walking Upstairs',
    2: 'Walking Downstairs',
    3: 'Sitting',
    4: 'Standing',
    5: 'Laying'
}

class SelfAttention(layers.Layer):
    """
    Self-Attention Layer for sequence data.
    Computes attention weights and applies them to create a context vector.
    """
    
    def __init__(self, units=64, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch, timesteps, features)
        
        # Compute attention scores
        # (batch, timesteps, features) @ (features, units) -> (batch, timesteps, units)
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[-1], [0]]) + self.b)
        
        # (batch, timesteps, units) @ (units,) -> (batch, timesteps)
        attention_weights = tf.tensordot(score, self.u, axes=[[-1], [0]])
        
        # Apply softmax to get normalized attention weights
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # Expand dims for broadcasting: (batch, timesteps) -> (batch, timesteps, 1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        # Apply attention weights to input
        # (batch, timesteps, features) * (batch, timesteps, 1) -> (batch, timesteps, features)
        weighted = x * attention_weights
        
        # Sum over timesteps to get context vector: (batch, features)
        context = tf.reduce_sum(weighted, axis=1)
        
        return context
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({'units': self.units})
        return config

def load_data():
    """Load preprocessed UCI HAR dataset"""
    print("Loading data...")
    try:
        X_train = np.load('assets/X_train_raw_scaled.npy')
        X_test = np.load('assets/X_test_raw_scaled.npy')
        y_train = np.load('assets/y_train_one_hot.npy')
        y_test = np.load('assets/y_test_one_hot.npy')
        
        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and show per-class accuracy"""
    print(f"\n{'='*50}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*50}")

    # Overall accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'-'*50}")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Per-class accuracy
    print("Per-class accuracy:")
    for i, label in ACTIVITY_LABELS.items():
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean() * 100
            print(f"  {label:20s}: {class_acc:6.2f}% ({mask.sum()} samples)")
            
    print(f"{'='*50}\n")
    return accuracy

def test_single_prediction(model, X_test, y_test, sample_idx=0):
    """Test prediction on a single sample"""
    sample = X_test[sample_idx:sample_idx+1]
    true_label = np.argmax(y_test[sample_idx])
    
    prediction = model.predict(sample, verbose=0)
    pred_label = np.argmax(prediction[0])
    confidence = prediction[0][pred_label] * 100
    
    print(f"Sample {sample_idx}:")
    print(f"  True:      {ACTIVITY_LABELS[true_label]}")
    print(f"  Predicted: {ACTIVITY_LABELS[pred_label]} ({confidence:.2f}%)")
    print(f"  Match:     {'✓' if pred_label == true_label else '✗'}")

def main():
    X_train, X_test, y_train, y_test = load_data()
    if X_test is None:
        return

    models_to_evaluate = [
        ('models/har_cnn_float32.keras', "Standard CNN (Float32)"),
        ('models/har_cnn_lstm_gru_attention_final.keras', "Hybrid CNN-LSTM-GRU Attention"),
        ('models/har_cnn_tinyml_final.keras', "TinyML CNN (Float32)"),
        ('models/har_cnn_gru_attention_tinyml_final.keras', "CNN-GRU TinyML (Float32)"),
    ]
    
    results = {}

    for model_path, name in models_to_evaluate:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        print(f"\nLoading {name} from {model_path}...")
        try:
            # Custom objects map for loading models with custom layers
            custom_objects = {'SelfAttention': SelfAttention}
            
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            
            try:
                # Limit line length to avoid console width issues
                model.summary(line_length=80) 
            except Exception:
                print("(Could not print model summary)")
                
            acc = evaluate_model(model, X_test, y_test, name)
            results[name] = acc
            
            # Test a few random predictions
            print("Random Sample Predictions:")
            np.random.seed(42)
            for idx in np.random.choice(len(X_test), 3, replace=False):
                test_single_prediction(model, X_test, y_test, idx)
                
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    print("\n" + "="*50)
    print("Final Model Comparison")
    print("="*50)
    for name, acc in results.items():
        print(f"{name:30s}: {acc*100:.2f}% Accuracy")
    print("="*50)

if __name__ == '__main__':
    main()
