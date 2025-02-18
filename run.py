# run.py
import streamlit as st
import tempfile
from src.audio import check_language_match

st.title("Audio Language Matching (MMS-LID)")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

# Step 2: Text input for the language code
st.markdown("Enter the **ISO code** (e.g., 'eng', 'fra', 'jpn', etc.) that you want to check against.")
language_text = st.text_input("Language code (ISO 639-3)")

if st.button("Check Language Match"):
    if uploaded_file is not None and language_text.strip():
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Call the function from audio.py without threshold
        result = check_language_match(tmp_file_path, language_text)
        
        # Display the result
        if result == "correct":
            st.success("Language Match: correct")
        else:
            st.error("Language Match: false")
    else:
        st.warning("Please upload a file and enter a language code.")
