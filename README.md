# 🎧 AI Voice Enhancer – Deep Learning Based Noise Reduction

## Problem Statement
Background noise significantly degrades speech quality in calls, recordings, and assistive technologies.
Traditional noise reduction methods struggle with real-world, non-stationary noise.
This project aims to enhance noisy speech signals using a deep learning–based approach.

## Why This Matters
- Clear speech is critical for communication, accessibility, and content creation
- Used in call centers, hearing aids, podcasts, and voice assistants
- Learning-based approaches outperform classical filters in complex environments

## Solution Overview
This project implements a U-Net–based speech enhancement model that learns to predict a magnitude mask
from noisy audio spectrograms. The predicted mask is applied to reconstruct cleaner speech signals.

Key ideas:
- Time–frequency masking
- Encoder–decoder with skip connections
- Data augmentation for robustness

## Features
- Synthetic mixing of clean speech with real environmental noise (SNR: 0–20 dB)
- U-Net architecture for spectrogram mask prediction
- Pitch and speed augmentation during training
- Mixed precision (AMP) for faster GPU training
- Early stopping and learning rate scheduling
- Generates enhanced audio samples for comparison
- Visualizes noisy vs enhanced spectrograms

## Results
The model demonstrates clear noise suppression while preserving speech structure.

**Artifacts generated:**
- `noisy.wav`
- `enhanced.wav`
- Spectrogram visualizations

(Attach waveform/spectrogram images here)

## Dataset
- Clean speech: <source or description>
- Noise samples: real environmental noise
- Sampling rate: 16 kHz
- Audio length: fixed-length segments
- Preprocessing: STFT magnitude extraction & normalization

## Model Architecture
| Component | Description |
|--------|-------------|
| Encoder | Extracts features from noisy magnitude spectrogram |
| Decoder | Reconstructs clean magnitude mask with skip connections |
| Output Activation | Softplus (non-negative masks) |
| Loss Function | 0.7 × MSE + 0.3 × L1 |

## How to Run
```bash
git clone <repo-url>
pip install -r requirements.txt
