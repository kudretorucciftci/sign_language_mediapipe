import os
import numpy as np
from tensorflow.keras.utils import to_categorical

data_path = "hand_coordinate_data"
X = []
y = []

letters = [chr(i) for i in range(65, 91) if i != 74]  # A-Z except J

for idx, letter in enumerate(letters):
    folder_path = os.path.join(data_path, letter)
    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = np.load(file_path)
                if data.shape == (63,):  # Expected shape
                    X.append(data)
                    y.append(idx)
            except Exception as e:
                print(f"Error: Could not load {file_path} -> {e}")

X = np.array(X)
y = np.array(y)

np.save("X_dataset.npy", X)
np.save("y_labels.npy", y)

print("✅ Dataset prepared successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)
