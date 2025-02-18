# audio.py
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from numpy.linalg import norm
from .model import tokenizer, model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize processor and model for audio
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

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

def get_audio_embedding(audio_path):
    audio, sr = preprocess_audio(audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = audio_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def check_language_match(audio_path, language_text, threshold=0.8):
    """
    Compare the audio embedding with the text embedding from the language model.
    Returns "correct" if above threshold, else "false".
    """
    # Audio embedding
    audio_embedding = get_audio_embedding(audio_path)
    
    # Text embedding (from XLM-RoBERTa or similar model)
    inputs = tokenizer(language_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # Request hidden states from the model
        lang_outputs = model(**inputs, output_hidden_states=True)
        # Use the last hidden state to compute an average embedding:
        language_embedding = lang_outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
    
    # Cosine similarity
    cos_sim = np.dot(audio_embedding, language_embedding) / (
        norm(audio_embedding) * norm(language_embedding)
    )
    return "correct" if cos_sim >= threshold else "false"

#if __name__ == '__main__':
    #test_audio_path = r"D:\All github projects\ForeignLanguagedetectionMLHAMweek\Japanese.mp3"  # Replace with a valid audio file path.
   # test_language = "japanese"
   # result = check_language_match(test_audio_path, test_language)
   ## print(f"Language match: {result}")