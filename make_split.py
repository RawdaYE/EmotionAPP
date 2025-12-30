import numpy as np
from sklearn.model_selection import train_test_split

y = np.load("data/labels.npy")

indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

np.save("data/train_idx.npy", train_idx)
np.save("data/test_idx.npy", test_idx)

print("Train/Test split saved")
