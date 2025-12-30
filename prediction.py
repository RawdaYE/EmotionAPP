import torch
import numpy as np
import joblib
from train_audio_cnn import AudioCNN
from train_mlp_fusion import FusionMLP

# ------------------------
# Config
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Number to emotion mapping
# ------------------------
num_to_emotion = {
    0: "neutral",
    1: "joy",
    2: "sadness",
    3: "anger",
    4: "fear",
    5: "surprise",
    6: "disgust"
}

# ------------------------
# Load models
# ------------------------
# Audio model
audio_model = AudioCNN(num_classes=7).to(DEVICE)
audio_model.load_state_dict(torch.load("models/audio_cnn.pt", map_location=DEVICE))
audio_model.eval()

# Fusion model
le = joblib.load("models/label_encoder.pkl")
input_dim = 128 + 256
num_classes = len(le.classes_)
fusion_model = FusionMLP(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
fusion_model.load_state_dict(torch.load("models/fusion_mlp.pt", map_location=DEVICE))
fusion_model.eval()

# ------------------------
# Prediction functions
# ------------------------
def predict_audio(audio_feat):
    with torch.no_grad():
        outputs = audio_model(audio_feat)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_prob = torch.softmax(outputs, dim=1)[0, pred_idx].item()
    return pred_idx, pred_prob


def predict_text(text_feat):
    # Dummy placeholder (as discussed)
    pred_idx = torch.randint(0, 7, (1,)).item()
    pred_prob = 0.50
    return pred_idx, pred_prob


def predict_fusion(audio_embedding, text_embedding):
    fusion_input = np.concatenate([audio_embedding, text_embedding], axis=0)
    fusion_input = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = fusion_model(fusion_input)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_prob = torch.softmax(outputs, dim=1)[0, pred_idx].item()

    return pred_idx, pred_prob


# ------------------------
# Demo run
# ------------------------
if __name__ == "__main__":

    print("\n" + "=" * 42)
    print(" Multimodal Emotion Recognition Demo ")
    print("=" * 42 + "\n")

    # Example inputs (preprocessed)
    audio_feat = torch.randn(1, 1, 64, 300).to(DEVICE)
    audio_embedding = np.load("data/audio_embeddings.npy")[0]
    text_embedding = np.load("data/text_svd_features.npy")[0]

    # Predictions
    audio_idx, audio_prob = predict_audio(audio_feat)
    text_idx, text_prob = predict_text(text_embedding)
    fusion_idx, fusion_prob = predict_fusion(audio_embedding, text_embedding)

    # Map labels
    audio_label = num_to_emotion[audio_idx]
    text_label = num_to_emotion[text_idx]
    fusion_label = num_to_emotion[fusion_idx]

    # Output
    print("Audio Prediction")
    print("-" * 20)
    print(f"Emotion    : {audio_label}")
    print(f"Confidence : {audio_prob:.2f}\n")

    print("Text Prediction")
    print("-" * 20)
    print(f"Emotion    : {text_label}")
    print(f"Confidence : {text_prob:.2f}\n")

    print("Fusion Prediction (Audio + Text)")
    print("-" * 32)
    print(f"Emotion    : {fusion_label}")
    print(f"Confidence : {fusion_prob:.2f}")

    print("\n" + "=" * 42)
