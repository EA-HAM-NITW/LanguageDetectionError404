import librosa
import numpy as np

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

import whisper

def get_predicted_language(audio_path: str) -> str:
    """
    Detect the language spoken in the audio file using OpenAI's Whisper tiny model.
    
    Args:
        audio_path (str): Path to the audio file.
    
    Returns:
        str: Detected language code (e.g., 'en' for English, 'fr' for French).
    """
    # Load the tiny model
    model = whisper.load_model("tiny")
    
    # Load and preprocess the audio file
    # whisper.load_audio loads the file and resamples it to 16kHz.
    audio = whisper.load_audio(audio_path)
    # Ensure the audio length is appropriate by padding or trimming it.
    audio = whisper.pad_or_trim(audio)
    
    # Transcribe the audio; this will also perform language detection.
    result = model.transcribe(audio, fp16=False)
    
    # Extract the language from the result dictionary.
    predicted_language = result.get("language", "unknown")
    
    return predicted_language

# Example usage:
if __name__ == "__main__":
    audio_file = r"C:\Users\Abhyuday Chauhan\PycharmProjects\LanguageDetectionError404\src\output\recording\recorded.wav"
    print("Predicted Language:", get_predicted_language(audio_file))


def check_language_match(audio_path, language_text):
    predicted_label = get_predicted_language(audio_path)
    print("Predicted label:", predicted_label)
    # Compare predicted_label (e.g., "eng") with your target language text
    return "correct" if predicted_label.lower() == language_text.lower() else "false"
