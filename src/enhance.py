import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.ndimage as ndi

from src.model import UNetMask

SR = 16000
N_FFT = 512
HOP_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetMask().to(DEVICE)
model.load_state_dict(torch.load("best_mask_unet.pth", map_location=DEVICE))
model.eval()


def enhance_audio(audio_path):

    wave, sr = librosa.load(audio_path, sr=SR)

    stft = librosa.stft(wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(stft), np.angle(stft)

    mag_norm = mag / (np.max(mag) + 1e-12)

    mag_tensor = torch.tensor(mag_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred_mask = model(mag_tensor).squeeze().cpu().numpy()

    pred_mask = ndi.gaussian_filter(pred_mask, sigma=1)

    pred_mask = np.clip(pred_mask, 0.05, 1.0)

    gamma = 1.3
    enhanced_mag = mag * (pred_mask ** gamma)

    enhanced_mag = np.maximum(enhanced_mag, 0.02 * mag)

    noise_floor = 0.001 * np.max(enhanced_mag)
    enhanced_mag[enhanced_mag < noise_floor] = 0.0

    frame_energy = enhanced_mag.mean(axis=0)
    speech_frames = frame_energy > 0.05 * frame_energy.max()

    enhanced_mag[:, ~speech_frames] *= 0.7
    enhanced_mag = np.maximum(enhanced_mag, 1e-6)

    enhanced_stft = enhanced_mag * np.exp(1j * phase)

    enhanced_wave = librosa.istft(enhanced_stft, hop_length=HOP_LENGTH)
    enhanced_wave = 0.95 * enhanced_wave / (np.max(np.abs(enhanced_wave)) + 1e-12)

    output_path = "enhanced_audio.wav"

    sf.write(output_path, enhanced_wave, SR)

    return output_path