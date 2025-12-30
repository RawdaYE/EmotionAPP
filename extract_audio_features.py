import torch
import numpy as np
import os

DATA_DIR = "data/preprocessed_data/train"
MAX_TIME = 300   # fixed length for mel spectrogram

X_audio = []
y = []

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

    mel = data["audio_mel"].numpy()   # shape: (1, 64, T)

    # Pad or truncate time dimension
    if mel.shape[2] < MAX_TIME:
        pad_width = MAX_TIME - mel.shape[2]
        mel = np.pad(mel, ((0,0),(0,0),(0,pad_width)))
    else:
        mel = mel[:, :, :MAX_TIME]

    X_audio.append(mel)
    y.append(emotion_map[data["emotion"]])

X_audio = np.array(X_audio)
y = np.array(y)

os.makedirs("data", exist_ok=True)
np.save("data/audio_features.npy", X_audio)
np.save("data/labels.npy", y)

print("Saved audio_features.npy and labels.npy")
print("Audio features shape:", X_audio.shape)
print("Labels shape:", y.shape)
