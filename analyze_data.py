import numpy as np
import os

files = [
    'assets/X_train_raw_scaled.npy',
    'assets/X_test_raw_scaled.npy',
    'assets/y_train_one_hot.npy',
    'assets/y_test_one_hot.npy'
]

print("Analysis of .npy files:\n")

for f in files:
    if os.path.exists(f):
        try:
            data = np.load(f)
            print(f"File: {f}")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min/Max: {data.min()}/{data.max()}")
            if 'y_' in f:
                 # Check classes
                 unique_rows = np.unique(data, axis=0)
                 print(f"  Unique rows (first 5): {unique_rows[:5]}")
                 print(f"  Number of unique classes: {len(unique_rows)}")
            print("-" * 30)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    else:
        print(f"File not found: {f}")
