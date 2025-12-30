import torch
import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = "data/preprocessed_data/train"

texts = []
labels = []

emotion_map = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "disgust": 6
}

files = os.listdir(DATA_DIR)

for file in files:
    if not file.endswith(".pt"):
        continue

    path = os.path.join(DATA_DIR, file)
    data = torch.load(path, weights_only=False)

    texts.append(data["utterance"])
    labels.append(emotion_map[data["emotion"]])

texts = np.array(texts)
labels = np.array(labels)

print("Text samples:", texts.shape)
print("Labels:", labels.shape)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_text,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Train text model
text_model = LogisticRegression(max_iter=1000)
text_model.fit(X_train, y_train)

# Evaluate
y_pred = text_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Text model accuracy:", acc)

# Save everything
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.save("data/text_features.npy", X_text.toarray())
joblib.dump(text_model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/text_vectorizer.pkl")

print("Text features and model saved")
