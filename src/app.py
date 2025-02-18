import io
import streamlit as st
from audiorecorder import audiorecorder  # Custom component for recording audio
from pydub import AudioSegment

from src.audio import check_language_match


def app():
    st.title("Audio Language Detection")

    # Button to start recording
    if st.button("Record Audio"):
        st.info("Click on the record button below to start recording.")
        
        # audiorecorder displays its own recording UI and returns audio data as bytes.
        audio_bytes = audiorecorder()
        
        if audio_bytes:
            st.success("Audio recorded successfully!")
            
            # Convert the recorded WAV bytes to MP3 using pydub.
            # Wrap the bytes in a BytesIO stream for pydub.
            wav_audio = io.BytesIO(audio_bytes)
            try:
                # Load the WAV audio; specify format if needed.
                audio_segment = AudioSegment.from_file(wav_audio, format="wav")
            except Exception as e:
                st.error("Error processing audio: " + str(e))
                st.stop()

            # Export audio as MP3 into a BytesIO object.
            mp3_io = io.BytesIO()
            audio_segment.export(mp3_io, format="mp3")
            mp3_bytes = mp3_io.getvalue()
            
            # Save the MP3 file to the project directory.
            mp3_filename = "recorded_audio.mp3"
            with open(mp3_filename, "wb") as f:
                f.write(mp3_bytes)
                
            # Play the recorded MP3 audio in Streamlit.
            st.audio(mp3_bytes, format="audio/mp3")
            
            # Pass the MP3 file path to your language detection model.
            detected_language = check_language_match(mp3_filename, language_text="Hello")
            st.write("Detected Language:", detected_language)
