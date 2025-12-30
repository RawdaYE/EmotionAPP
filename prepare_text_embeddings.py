import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib

# Load TF-IDF features (already created by train_text.py)
X_text = np.load("data/text_features.npy")   # shape: (N, vocab_size)

print("Original TF-IDF shape:", X_text.shape)

# Apply SVD (LSA)
svd = TruncatedSVD(n_components=256, random_state=42)
X_text_svd = svd.fit_transform(X_text)

print("Reduced text shape:", X_text_svd.shape)

# Save
np.save("data/text_svd_features.npy", X_text_svd)
joblib.dump(svd, "models/text_svd.pkl")

print("✅ Text SVD features saved")
