import torch
import joblib
import numpy as np

from train_audio_cnn import AudioCNN  # uses your trained CNN class

# ------------------------
# Config
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PT_FILE = "data/preprocessed_data/test/dia0_utt0.pt"

# ------------------------
# Load label encoder
# ------------------------
label_encoder = joblib.load("models/label_encoder.pkl")

# ------------------------
# Load audio CNN
# ------------------------
num_classes = len(label_encoder.classes_)
audio_model = AudioCNN(num_classes)
audio_model.load_state_dict(torch.load("models/audio_cnn.pt", map_location=DEVICE))
audio_model.to(DEVICE)
audio_model.eval()

# ------------------------
# Load ONE preprocessed sample
# ------------------------
data = torch.load(PT_FILE, weights_only=False)

# Extract mel spectrogram
mel = data["audio_mel"]            # shape: (1, 64, T)
mel = mel.unsqueeze(0).to(DEVICE)  # (1, 1, 64, T)

# ------------------------
# Predict
# ------------------------
with torch.no_grad():
    outputs = audio_model(mel)
    probs = torch.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

emotion = label_encoder.inverse_transform([pred_idx])[0]

print(f"Audio Emotion: {emotion} ({confidence:.2f})")
