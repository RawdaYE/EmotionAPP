import joblib
import numpy as np

# ------------------------
# Load models
# ------------------------
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ------------------------
# Prediction function
# ------------------------
def predict_text(text):
    # Vectorize text
    X = text_vectorizer.transform([text])

    # Predict class
    pred_idx = text_model.predict(X)[0]

    # Predict confidence (if supported)
    if hasattr(text_model, "predict_proba"):
        confidence = np.max(text_model.predict_proba(X))
    else:
        confidence = None

    # Decode label
    emotion = label_encoder.inverse_transform([pred_idx])[0]

    return emotion, confidence


# ------------------------
# Run example
# ------------------------
if __name__ == "__main__":
    text_input = "I am so happy today!"

    emotion, confidence = predict_text(text_input)

    if confidence is not None:
        print(f"Text Emotion: {emotion} ({confidence:.2f})")
    else:
        print(f"Text Emotion: {emotion}")
