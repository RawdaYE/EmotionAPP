import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load features
X_audio = np.load("data/audio_embeddings.npy")      # (N, 128)
X_text = np.load("data/text_features.npy")          # (N, D)
y = np.load("data/labels.npy")

# Combine features
X = np.hstack([X_audio, X_text])

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fusion classifier
fusion_model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1
)

fusion_model.fit(X_train, y_train)

# Evaluate
y_pred = fusion_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Multimodal Fusion accuracy:", acc)

# Save
joblib.dump(fusion_model, "models/fusion_model.pkl")
print("Fusion model saved")
