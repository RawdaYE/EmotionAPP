import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)

# -----------------------------
# Load features
# -----------------------------
X_audio = np.load("data/audio_embeddings.npy")      # (N, 128)
X_text = np.load("data/text_svd_features.npy")      # (N, 256)
y = np.load("data/labels.npy")

# 🔥 Make audio louder
X_audio = X_audio * 1.5

# Concatenate
X = np.concatenate([X_audio, X_text], axis=1)       # (N, 384)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, "models/label_encoder.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Dataset
# -----------------------------
class FusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# MLP Fusion Model
# -----------------------------
class FusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Training script
# -----------------------------
if __name__ == "__main__":
    train_loader = DataLoader(FusionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FusionDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = FusionMLP(input_dim=X.shape[1], num_classes=len(np.unique(y))).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())

    acc = accuracy_score(y_true, y_pred)
    print("MLP Fusion accuracy:", acc)

    # Save
    torch.save(model.state_dict(), "models/fusion_mlp.pt")
    print("Fusion MLP saved")
