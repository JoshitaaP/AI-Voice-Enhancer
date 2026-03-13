# AI Voice Enhancer
A deep learning–based speech enhancement system that reduces background noise using a UNet neural network and spectrogram masking.
The model predicts a speech mask over a noisy spectrogram and reconstructs cleaner audio using Inverse Short-Time Fourier Transform (ISTFT).

## Project Overview
Background noise often reduces the quality and intelligibility of recorded speech.
This project enhances speech by applying deep learning techniques to suppress noise while preserving important speech frequencies.
The system works by:
1. Converting audio signals into spectrograms
2. Feeding the spectrogram into a UNet neural network
3. Predicting a speech enhancement mask
4. Applying the mask to suppress noise
6. Reconstructing enhanced audio using ISTFT

## Model Architecture
The model uses a UNet-style encoder-decoder architecture designed for spectrogram enhancement.
### Encoder
Extracts hierarchical audio features from the noisy spectrogram.

Input Spectrogram  
↓  
Conv + BatchNorm + ReLU  
↓  
Downsampling  
↓  
Feature Extraction

### Decoder
Upsamples the features and reconstructs the enhanced spectrogram using skip connections.

Upsampling  
↓  
Skip Connections  
↓  
Mask Prediction

The final output is a speech mask applied to the noisy spectrogram.

## Enhancement Pipeline
Noisy Audio  
↓  
Short-Time Fourier Transform (STFT)  
↓  
Spectrogram Magnitude  
↓  
UNet Mask Prediction  
↓  
Mask Application  
↓  
Inverse STFT  
↓  
Enhanced Audio

## Results
The model improves speech clarity by suppressing noise in the frequency domain.
Example visualization:

Noisy Spectrogram → Predicted Mask → Enhanced Spectrogram

You can generate these visualizations using the notebook in:
`notebooks/audio_enhancement.ipynb`

## Streamlit Demo
This project includes a Streamlit web app where users can upload noisy audio and listen to the enhanced result.
### Run the app locally
Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Then open:

http://localhost:8501

Upload a noisy audio file and listen to the enhanced version.

## Tech Stack
- Python
-  PyTorch
- Librosa
- NumPy
- Matplotlib
- Streamlit

## Repository Structure

```
AI-Voice-Enhancer
│
├── notebooks
│   └── audio_enhancement.ipynb
│
├── src
│   ├── model.py     
│   └── enhance.py      
│
├── app.py              
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements
- Real-time audio enhancement
- Better noise suppression in extremely noisy environments
