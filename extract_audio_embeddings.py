import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from train_audio_cnn import AudioCNN
import joblib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load("data/audio_features.npy")
y = np.load("data/labels.npy")

class AudioDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

loader = DataLoader(AudioDataset(X), batch_size=32, shuffle=False)

num_classes = len(np.unique(y))
model = AudioCNN(num_classes)
model.load_state_dict(torch.load("models/audio_cnn.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

embeddings = []

with torch.no_grad():
    for xb in loader:
        xb = xb.to(DEVICE)
        emb = model(xb, return_embedding=True)
        embeddings.append(emb.cpu().numpy())

audio_embeddings = np.vstack(embeddings)

np.save("data/audio_embeddings.npy", audio_embeddings)
print("Saved audio embeddings shape:", audio_embeddings.shape)
