import os
import streamlit as st
from src.enhance import enhance_audio

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Voice Enhancer",
    page_icon="🎤",
    layout="centered"
)

st.title(" AI Voice Enhancer")
st.write(
    "Upload a noisy speech recording and enhance it using the trained U-Net Speech Enhancement model."
)

# -----------------------------
# Upload Audio
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac"]
)

if uploaded_file is not None:

    # Preserve original extension
    extension = os.path.splitext(uploaded_file.name)[1]
    input_path = "input_audio" + extension

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Audio uploaded successfully!")

    st.subheader("🎵 Original Audio")
    st.audio(input_path)

    if st.button("Enhance Audio"):

        with st.spinner("Enhancing audio... Please wait."):
            enhanced_path = enhance_audio(input_path)

        st.success("Enhancement Completed!")

        st.subheader("🎧 Enhanced Audio")
        st.audio(enhanced_path)

        with open(enhanced_path, "rb") as audio_file:
            st.download_button(
                label="⬇ Download Enhanced Audio",
                data=audio_file,
                file_name="enhanced_audio.wav",
                mime="audio/wav"
            )