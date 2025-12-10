# 🎧 AI Voice Enhancer – U-Net Based Noise Reduction

A deep-learning project that removes background noise from speech using a U-Net based mask prediction model.  
The network learns to generate a magnitude mask from noisy spectrograms and reconstructs enhanced audio from it.  
This repository is in active development — more features & improvements will be added regularly.

---

## 🔥 Features
- Mixes clean speech with real environmental noise (random SNR: 0–20dB)
- U-Net architecture for mask prediction and spectrogram enhancement
- Random pitch & speed augmentations for robust training
- Mixed Precision (AMP) support for faster GPU training
- Early stopping + LR scheduling for stable convergence
- Generates `noisy.wav` & `enhanced.wav` for comparison
- Visualizes noisy vs mask vs enhanced spectrograms

---

## 🧠 Model Overview

Architecture Highlights:

| Component | Purpose |
|---|---|
| Encoder | Feature extraction from noisy magnitude |
| Decoder | Reconstruct clean mask with skip connections |
| Softplus Output | Ensures smooth non-negative masks |
| Loss Function | 0.7 * MSE + 0.3 * L1 |

---

## 🔧 Tech Stack

| Used For | Technology |
|---|---|
| Model Training | PyTorch |
| Audio Processing | Librosa, SoundFile |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab (GPU) |

---
