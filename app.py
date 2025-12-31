import os
import torch
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
import tempfile

from train_audio_cnn import AudioCNN
from train_mlp_fusion import FusionMLP

# ------------------------
# Config
# ------------------------
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TIME = 300  # Fixed time steps for mel spectrogram
N_MELS = 64  # Number of mel filter banks
NUM_TO_EMOTION = {
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
print("Loading models...")

try:
    # Load label encoder
    if not os.path.exists("models/label_encoder.pkl"):
        raise FileNotFoundError("models/label_encoder.pkl not found. Please train the models first.")
    label_encoder = joblib.load("models/label_encoder.pkl")

    # Load audio CNN model
    if not os.path.exists("models/audio_cnn.pt"):
        raise FileNotFoundError("models/audio_cnn.pt not found. Please train the models first.")
    num_classes = len(label_encoder.classes_)
    audio_model = AudioCNN(num_classes).to(DEVICE)
    audio_model.load_state_dict(torch.load("models/audio_cnn.pt", map_location=DEVICE))
    audio_model.eval()
    print("[+] Audio CNN model loaded")

    # Load text model and vectorizer
    if not os.path.exists("models/text_model.pkl"):
        raise FileNotFoundError("models/text_model.pkl not found. Please train the models first.")
    if not os.path.exists("models/text_vectorizer.pkl"):
        raise FileNotFoundError("models/text_vectorizer.pkl not found. Please train the models first.")
    text_model = joblib.load("models/text_model.pkl")
    text_vectorizer = joblib.load("models/text_vectorizer.pkl")
    print("[+] Text model loaded")

    # Load text SVD for fusion
    if not os.path.exists("models/text_svd.pkl"):
        raise FileNotFoundError("models/text_svd.pkl not found. Please train the models first.")
    text_svd = joblib.load("models/text_svd.pkl")
    print("[+] Text SVD loaded")

    # Load fusion model
    if not os.path.exists("models/fusion_mlp.pt"):
        raise FileNotFoundError("models/fusion_mlp.pt not found. Please train the models first.")
    input_dim = 128 + 256  # audio embeddings + text SVD features
    fusion_model = FusionMLP(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    fusion_model.load_state_dict(torch.load("models/fusion_mlp.pt", map_location=DEVICE))
    fusion_model.eval()
    print("[+] Fusion model loaded")

    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please ensure all model files are present in the models/ directory.")
    print("You may need to run the training scripts first.")
    raise
except Exception as e:
    print(f"ERROR loading models: {e}")
    raise
# ------------------------
# Audio preprocessing
# ------------------------
def audio_to_melspectrogram(audio_path, sr=22050, n_mels=64, max_time=300):
    """
    Convert audio file to mel spectrogram.
    Returns: mel spectrogram of shape (1, n_mels, max_time)
    """
    try:
        # Load audio file
        y, original_sr = librosa.load(audio_path, sr=sr, mono=True)
        
        # Check if audio is too short
        if len(y) < sr * 0.1:  # Less than 0.1 seconds
            raise ValueError("Audio file is too short (less than 0.1 seconds)")
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
            fmax=sr/2
        )
        
        # Convert to log scale (dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Reshape to (1, n_mels, time_steps)
        mel_db = mel_db.reshape(1, n_mels, -1)
        
        # Pad or truncate time dimension
        current_len = mel_db.shape[2]
        if current_len < max_time:
            pad_width = max_time - current_len
            mel_db = np.pad(mel_db, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            mel_db = mel_db[:, :, :max_time]
        
        return mel_db
    
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

# ------------------------
# Prediction functions
# ------------------------
def predict_audio_from_file(audio_path):
    """Predict emotion from audio file."""
    # Convert audio to mel spectrogram
    mel = audio_to_melspectrogram(audio_path)
    
    # Create raw tensor
    mel_tensor_raw = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 1. For Audio Prediction: Use Instance Normalization
    # The AudioCNN was trained with (X - X.mean()) / X.std() in the Dataset class.
    # So we must do the same here to get accurate audio predictions.
    mel_mean = mel_tensor_raw.mean()
    mel_std = mel_tensor_raw.std() + 1e-6
    mel_tensor_norm = (mel_tensor_raw - mel_mean) / mel_std
    
    # Predict Audio Emotion
    with torch.no_grad():
        # Pass instance-normalized data for correct classification
        outputs = audio_model(mel_tensor_norm)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    label_str = label_encoder.inverse_transform([pred_idx])[0]
    # If label_encoder returns an integer or numeric string, try to map it
    try:
        emotion = NUM_TO_EMOTION[int(label_str)]
    except (ValueError, KeyError):
        emotion = str(label_str)
    
    # 2. For Fusion Embedding: Use Statistical Matching
    # The Fusion model was trained on embeddings generated from 'raw' data (accidentally).
    # That 'raw' data had a specific distribution (Mean ~ -3.39, Std ~ 3.34).
    # To get consistent embeddings, we must force our input to match that distribution.
    # We take our normalized data (Mean 0, Std 1) and scale/shift it to match training stats.
    GLOBAL_MEAN = -3.3936
    GLOBAL_STD = 3.3429
    mel_tensor_matched = mel_tensor_norm * GLOBAL_STD + GLOBAL_MEAN

    with torch.no_grad():
        # Pass stat-matched data to get the expected 'garbage' embedding
        audio_embedding = audio_model(mel_tensor_matched, return_embedding=True)
        audio_embedding = audio_embedding.cpu().numpy()[0]
        # Apply same scaling as training
        audio_embedding = audio_embedding * 1.5
    
    return str(emotion), float(confidence), audio_embedding

def predict_text_from_string(text):
    """Predict emotion from text string."""
    # Vectorize text
    X = text_vectorizer.transform([text])
    
    # Predict class
    pred_idx = text_model.predict(X)[0]
    
    # Predict confidence
    if hasattr(text_model, "predict_proba"):
        confidence = np.max(text_model.predict_proba(X))
    else:
        confidence = 0.5
    
    # Decode label
    label_str = label_encoder.inverse_transform([pred_idx])[0]
    try:
        emotion = NUM_TO_EMOTION[int(label_str)]
    except (ValueError, KeyError):
        emotion = str(label_str)
    
    # Get text embedding for fusion (apply SVD)
    text_features = X.toarray()
    text_embedding = text_svd.transform(text_features)[0]
    
    return str(emotion), float(confidence), text_embedding
def predict_fusion_from_embeddings(audio_embedding, text_embedding):
    """Predict emotion using fusion model."""
    # Concatenate features
    fusion_input = np.concatenate([audio_embedding, text_embedding], axis=0)
    fusion_input = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = fusion_model(fusion_input)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    label_str = label_encoder.inverse_transform([pred_idx])[0]
    try:
        emotion = NUM_TO_EMOTION[int(label_str)]
    except (ValueError, KeyError):
        emotion = str(label_str)
    
    return str(emotion), float(confidence)

# ------------------------
# API Routes
# ------------------------
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/predict/audio', methods=['POST'])
def predict_audio():
    """Predict emotion from audio file."""
    temp_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format. Supported formats: {", ".join(allowed_extensions)}'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        print(f"Processing audio file: {filename}")
        
        try:
            # Predict
            emotion, confidence, audio_embedding = predict_audio_from_file(temp_path)
            
            print(f"Prediction successful: {emotion} ({confidence:.2f})")
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': float(confidence),
                'audio_embedding': audio_embedding.tolist()
            })
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    except Exception as e:
        print(f"Error in predict_audio endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/text', methods=['POST'])
def predict_text():
    """Predict emotion from text."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Text is empty'}), 400
        
        print(f"Processing text: {text[:50]}...")
        
        # Predict
        emotion, confidence, text_embedding = predict_text_from_string(text)
        
        print(f"Prediction successful: {emotion} ({confidence:.2f})")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': float(confidence),
            'text_embedding': text_embedding.tolist()
        })
    
    except Exception as e:
        print(f"Error in predict_text endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/fusion', methods=['POST'])
def predict_fusion():
    """Predict emotion using fusion model (requires both audio and text)."""
    try:
        data = request.get_json()
        if 'audio_embedding' not in data or 'text_embedding' not in data:
            return jsonify({'error': 'Both audio_embedding and text_embedding required'}), 400
        
        audio_embedding = np.array(data['audio_embedding'])
        text_embedding = np.array(data['text_embedding'])
        
        # Predict
        emotion, confidence = predict_fusion_from_embeddings(audio_embedding, text_embedding)
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/multimodal', methods=['POST'])
def predict_multimodal():
    """Predict emotion from both audio file and text in one request."""
    temp_path = None
    try:
        # Get text from form data
        text = request.form.get('text', '')
        
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not text.strip():
            return jsonify({'error': 'Text is required for multimodal prediction'}), 400
        
        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format. Supported formats: {", ".join(allowed_extensions)}'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        print(f"Processing multimodal: audio={filename}, text={text[:50]}...")
        
        try:
            # Get predictions and embeddings
            audio_emotion, audio_conf, audio_emb = predict_audio_from_file(temp_path)
            text_emotion, text_conf, text_emb = predict_text_from_string(text)
            fusion_emotion, fusion_conf = predict_fusion_from_embeddings(audio_emb, text_emb)
            
            print(f"Multimodal prediction successful - Audio: {audio_emotion}, Text: {text_emotion}, Fusion: {fusion_emotion}")
            
            return jsonify({
                'success': True,
                'audio': {
                    'emotion': audio_emotion,
                    'confidence': float(audio_conf)
                },
                'text': {
                    'emotion': text_emotion,
                    'confidence': float(text_conf)
                },
                'fusion': {
                    'emotion': fusion_emotion,
                    'confidence': float(fusion_conf)
                }
            })
        except Exception as e:
            print(f"Error processing multimodal: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing multimodal: {str(e)}'}), 500
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    except Exception as e:
        print(f"Error in predict_multimodal endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)