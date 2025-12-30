import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
# Load prepared audio features and labels
X_audio = np.load("data/audio_features.npy")   # shape: (N, 1, 64, 300)
y = np.load("data/labels.npy")

# Flatten audio features
X_audio = X_audio.reshape(X_audio.shape[0], -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_audio, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
audio_model = LogisticRegression(
    max_iter=1000 
)
audio_model.fit(X_train, y_train)

# Evaluate
y_pred = audio_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Audio model accuracy:", acc)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(audio_model, "models/audio_model.pkl")
print("Audio model saved")
