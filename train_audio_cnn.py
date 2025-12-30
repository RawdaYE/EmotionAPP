import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# ------------------------
# Config
# ------------------------
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)

# ------------------------
# Dataset
# ------------------------
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X = (self.X - self.X.mean()) / (self.X.std() + 1e-6)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------
# CNN Model
# ------------------------
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if return_embedding:
            return x  # 128-d embedding

        return self.classifier(x)

# ============================================================
# 🔥 MAIN (TRAINING RUNS ONLY IF FILE IS EXECUTED DIRECTLY)
# ============================================================
if __name__ == "__main__":

    # ------------------------
    # Load data
    # ------------------------
    X = np.load("data/audio_features.npy")   # (N, 1, 64, 300)
    y = np.load("data/labels.npy")

    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, "models/label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = DataLoader(
        AudioDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        AudioDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    num_classes = len(np.unique(y))
    model = AudioCNN(num_classes).to(DEVICE)

    # ------------------------
    # Training
    # ------------------------
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

    # ------------------------
    # Evaluation
    # ------------------------
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
    print("Audio CNN accuracy:", acc)

    # ------------------------
    # Save
    # ------------------------
    torch.save(model.state_dict(), "models/audio_cnn.pt")
    print("Audio CNN model saved")
