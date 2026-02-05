
import matplotlib
matplotlib.use("TkAgg")  # explicitly non-GUI

import matplotlib.pyplot as plt
import numpy as np

X = np.load("assets/X_train_raw_scaled.npy")

sample = X[0]

plt.figure(figsize=(12, 6))
for i in range(9):
    plt.plot(sample[:, i])

plt.title("Sensor signals")
plt.xlabel("Time step")
plt.ylabel("Value")

plt.savefig("sample_signals.png", dpi=150, bbox_inches="tight")
plt.close()
