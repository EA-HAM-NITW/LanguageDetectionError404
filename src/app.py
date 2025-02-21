import streamlit as st

from src.sound import sound
from src.settings import WAVE_OUTPUT_FILE,DURATION
from src.audio import check_language_match  # Function that processes the audio file


GROUND_TRUTH = "en"  # Update this value as needed


def reset_app():
    # Clear session state if any variables were stored (optional)
    for key in st.session_state.keys():
        del st.session_state[key]
    # Rerun the app, which resets all UI elements to their initial state
    st.experimental_rerun()

def app():
    # Wrap the entire app in a container
    with st.container():
        st.title("Audio Language Detection")

        # Place the Reset button at the top of the container

        # Ground truth variable for language detection

        # Button to start recording
        if st.button('Record'):
            with st.spinner(f"Recording for {DURATION} seconds ..."):
                sound.record()  # Record audio and save to WAVE_OUTPUT_FILE
            st.success("Recording completed")

            print('A')
            # Call check_language_match with the recorded file and the ground truth language
            detected_language = check_language_match(WAVE_OUTPUT_FILE, language_text=GROUND_TRUTH)
            print('B')

            # Output section
            st.subheader("Output:")
            if detected_language == GROUND_TRUTH.lower():
                st.success("Valid")
            else:
                st.error("Invalid")
            
            st.write("Detected Language:", detected_language)
                
            if st.button("Reset App"):
                reset_app()

