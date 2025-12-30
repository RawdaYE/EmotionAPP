import torch
import numpy as np
import os

train_path = "data/preprocessed_data/train"
X_audio = []
y = []

emotion_map = {
    'neutral':0, 'joy':1, 'sadness':2, 'anger':3, 'fear':4, 'disgust':5, 'surprise':6
}

max_len = 300  # desired time_steps

for file_name in os.listdir(train_path):
    if file_name.endswith(".pt"):
        file_path = os.path.join(train_path, file_name)
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
            data = torch.load(file_path, weights_only=False)
        
        mel = data['audio_mel'].numpy()  # shape: (1, num_mels, time_steps)
        
        # Pad or truncate last dimension (time_steps)
        current_len = mel.shape[2]
        if current_len < max_len:
            pad_width = max_len - current_len
            # pad only the last axis
            mel = np.pad(mel, ((0,0),(0,0),(0,pad_width)), mode='constant')
        else:
            mel = mel[:, :, :max_len]
        
        X_audio.append(mel)
        
        # Emotion label
        emotion_label = data['emotion']
        y.append(emotion_map[emotion_label])

# Convert to numpy arrays
X_audio = np.array(X_audio)  # shape: (num_samples, 1, num_mels, max_len)
y = np.array(y)

print("Audio features shape:", X_audio.shape)
print("Labels shape:", y.shape)
