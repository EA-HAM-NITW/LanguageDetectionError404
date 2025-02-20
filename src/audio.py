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


def get_predicted_language(audio_path: str) -> str:
    """
    Predict the language of the audio in the given file.

    Args:
        audio_path (str): Path to the audio file (preferably a mono WAV at 16 kHz).

    Returns:
        str: The predicted language label.
    """ 
    import whisper
    model = whisper.load_model('tiny')

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Transcribe the audio; this will also perform language detection.
    result = model.transcribe(audio, fp16=False)
    predicted_language=result['language']
    print(predicted_language)
    return predicted_language

def check_language_match(audio_path, language_text):
    predicted_label = get_predicted_language(audio_path)
    print("Predicted label:", predicted_label)
    # Compare predicted_label (e.g., "eng") with your target language text
    return "correct" if predicted_label.lower() == language_text.lower() else "false"


if __name__ == "__main__":
    audio_path = "data/test.wav"
    location = r"C:\Users\Abhyuday Chauhan\PycharmProjects\LanguageDetectionError404\src\output\recording\recorded.wav"
    import os
    print(os.path.exists(location))
    print(check_language_match(location, 'en'))