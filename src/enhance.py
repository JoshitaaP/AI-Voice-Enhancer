import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.ndimage as ndi
import scipy.signal as sig

from src.model import UNetMask

SR = 16000
N_FFT = 512
HOP_LENGTH = 256   # MUST match training (train.py). Do not change independently
                   # of train.py's HOP_LENGTH — mismatched hop length feeds the
                   # model spectrograms at a time-resolution it never trained on.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetMask().to(DEVICE)
model.load_state_dict(torch.load("best_mask_unet.pth", map_location=DEVICE))
model.eval()


def _estimate_noise_profile(mag, pct_range=(10, 30)):
    """Estimate the noise spectrum from this clip's own low-energy (but non-
    silent) frames, as a light safety net on top of the mask."""
    frame_energy = mag.mean(axis=0)
    nonzero = frame_energy > 1e-6
    if nonzero.sum() < 5:
        return np.zeros((mag.shape[0], 1))
    lo, hi = np.percentile(frame_energy[nonzero], pct_range)
    band_idx = np.where(nonzero & (frame_energy >= lo) & (frame_energy <= hi))[0]
    if len(band_idx) == 0:
        return np.zeros((mag.shape[0], 1))
    return mag[:, band_idx].mean(axis=1, keepdims=True)


def _spectral_subtract(mag, noise_profile, oversubtraction=0.5, spectral_floor=0.1):
    subtracted = mag - oversubtraction * noise_profile
    floor = spectral_floor * mag
    return np.maximum(subtracted, floor)


def _highpass(wave, sr, cutoff=80.0):
    b, a = sig.butter(2, cutoff / (sr / 2), btype="highpass")
    return sig.filtfilt(b, a, wave)


def _loudness_normalize(wave, target_rms=0.09, limiter_ceiling=0.97):
    rms = np.sqrt(np.mean(wave ** 2)) + 1e-12
    wave = wave * (target_rms / rms)
    peak = np.max(np.abs(wave)) + 1e-12
    if peak > limiter_ceiling:
        wave = np.tanh(wave / peak * 1.5) * limiter_ceiling
    return wave


def enhance_audio(audio_path):

    # -----------------------------
    # Load Audio
    # -----------------------------
    wave, sr = librosa.load(audio_path, sr=SR)
    original_len = len(wave)

    # -----------------------------
    # STFT
    # -----------------------------
    stft = librosa.stft(
        wave,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mag = np.abs(stft)
    phase = np.angle(stft)

    # -----------------------------
    # Same preprocessing as training
    # -----------------------------
    mag_log = np.log1p(mag)
    mag_norm = mag_log / (mag_log.max() + 1e-8)

    mag_tensor = (
        torch.tensor(mag_norm)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    # -----------------------------
    # Predict Mask
    # -----------------------------
    with torch.no_grad():
        pred_mask = model(mag_tensor).squeeze().cpu().numpy()

    pred_mask = ndi.gaussian_filter1d(
        pred_mask,
        sigma=1.0,
        axis=1
    )

    pred_mask = np.clip(pred_mask, 0.05, 1.0)

    # -----------------------------
    # Apply Mask
    # -----------------------------
    masked_mag = mag * pred_mask

    noise_profile = _estimate_noise_profile(mag)

    enhanced_mag = _spectral_subtract(
        masked_mag,
        noise_profile,
        oversubtraction=0.5,
        spectral_floor=0.1
    )

    # -----------------------------
    # Reconstruct Speech
    # -----------------------------
    enhanced_stft = enhanced_mag * np.exp(1j * phase)

    enhanced_wave = librosa.istft(
        enhanced_stft,
        hop_length=HOP_LENGTH,
        length=original_len
    )

    # =====================================================
    # Save RAW output (used for evaluation metrics)
    # =====================================================

    raw_output = "enhanced_raw.wav"

    sf.write(
        raw_output,
        enhanced_wave,
        SR
    )

    # =====================================================
    # Post-processing (used for demo only)
    # =====================================================

    enhanced_demo = _highpass(
        enhanced_wave,
        SR,
        cutoff=80
    )

    enhanced_demo = _loudness_normalize(
        enhanced_demo,
        target_rms=0.09,
        limiter_ceiling=0.97
    )

    output_path = "enhanced_audio.wav"

    sf.write(
        output_path,
        enhanced_demo,
        SR
    )

    return output_path
