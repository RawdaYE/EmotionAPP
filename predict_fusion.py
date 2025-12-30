import torch
import numpy as np
import joblib
import os

# ------------------------
# Config
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load models
# ------------------------
from train_mlp_fusion import FusionMLP  # import the model class

# Load label encoder
le = joblib.load("models/label_encoder.pkl")

# Load trained Fusion MLP
input_dim = 128 + 256  # audio embeddings + text SVD features
num_classes = len(le.classes_)
fusion_model = FusionMLP(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
fusion_model.load_state_dict(torch.load("models/fusion_mlp.pt", map_location=DEVICE))
fusion_model.eval()

# ------------------------
# Prediction function
# ------------------------
def predict_fusion(audio_embedding, text_embedding):
    """
    audio_embedding: np.array, shape (128,)
    text_embedding: np.array, shape (256,)
    """
    # Concatenate features
    fusion_input = np.concatenate([audio_embedding, text_embedding], axis=0)
    fusion_input = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = fusion_model(fusion_input)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_label = le.inverse_transform([pred_idx])[0]
        pred_prob = torch.softmax(outputs, dim=1)[0, pred_idx].item()

    return pred_label, pred_prob

# ------------------------
# Example run
# ------------------------
if __name__ == "__main__":
    # Load preprocessed embeddings
    audio_embedding = np.load("data/audio_embeddings.npy")[0]  # example: first audio sample
    text_embedding = np.load("data/text_svd_features.npy")[0]  # example: first text sample

    emotion, prob = predict_fusion(audio_embedding, text_embedding)
    print(f"Predicted Emotion: {emotion} ({prob:.2f})")
