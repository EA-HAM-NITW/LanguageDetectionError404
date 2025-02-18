# audio.py
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-4017")
model = AutoModelForAudioClassification.from_pretrained("facebook/mms-lid-4017")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def preprocess_audio(audio_path, target_sr=16000, target_duration=60):
    y, sr = librosa.load(audio_path, sr=target_sr)
    y = librosa.util.normalize(y)
    target_length = target_sr * target_duration
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    else:
        y = y[:target_length]
    return y, sr

def get_predicted_language(audio_path):
    audio_array, sr = preprocess_audio(audio_path)
    inputs = feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_id = logits.argmax(dim=-1).item()
        predicted_label = model.config.id2label[predicted_id]
    return predicted_label

def check_language_match(audio_path, language_text):
    predicted_label = get_predicted_language(audio_path)
    print("Predicted label:", predicted_label)
    # Compare predicted_label (e.g., "eng") with your target language text
    return "correct" if predicted_label.lower() == language_text.lower() else "false"