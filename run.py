# run.py
import streamlit as st
import tempfile
import os 
import sys
from src.audio import check_language_match

st.title("Audio Language Matching")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

# Step 2: Text input for the language
language_text = st.text_input("Enter or select the language text to compare against")

# Step 3: Threshold slider (optional, to experiment with thresholds)
threshold = st.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.8, 0.01)

if st.button("Check Language Match"):
    if uploaded_file is not None and language_text.strip():
        # Streamlit uploads files in-memory, so we need to temporarily save it to disk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Call your function from audio.py
        result = check_language_match(tmp_file_path, language_text, threshold=threshold)
        
        # Step 4: Display the result
        if result == "correct":
            st.success(f"Language Match: {result}")
        else:
            st.error(f"Language Match: {result}")
    else:
        st.warning("Please upload a file and enter language text.")
