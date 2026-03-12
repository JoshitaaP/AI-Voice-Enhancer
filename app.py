import streamlit as st
from src.enhance import enhance_audio

st.title("AI Voice Enhancer")

st.write("Upload a noisy audio file and enhance it using a UNet model.")


uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav","mp3","flac"]
)

if uploaded_file is not None:

    with open("input_audio.wav","wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Original Audio")
    st.audio("input_audio.wav")

    enhanced = enhance_audio("input_audio.wav")

    st.subheader("Enhanced Audio")
    st.audio(enhanced)