"""
Train the U-Net speech-mask model on the VoiceBank-DEMAND dataset
(real recorded noisy/clean pairs, not synthetic mixing).

Just edit the four paths in CONFIG below, then run:
    python train.py

Saves the best checkpoint to ../best_mask_unet.pth (same location enhance.py
loads from) whenever validation loss improves.
"""

import os
import random
import glob

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import UNetMask

# ============================== CONFIG ======================================
# EDIT THESE FOUR PATHS FOR YOUR MACHINE (point at the extracted zip folders)
NOISY_TRAIN_DIR = r"C:\Users\JOSHITAA\Documents\404FunNotFound\AI-Voice-Enhancer\data\archive\noisy_trainset_28spk_wav"
CLEAN_TRAIN_DIR = r"C:\Users\JOSHITAA\Documents\404FunNotFound\AI-Voice-Enhancer\data\archive\clean_trainset_28spk_wav"

NOISY_TEST_DIR = r"C:\Users\JOSHITAA\Documents\404FunNotFound\AI-Voice-Enhancer\data\archive\noisy_testset_wav"
CLEAN_TEST_DIR = r"C:\Users\JOSHITAA\Documents\404FunNotFound\AI-Voice-Enhancer\data\archive\clean_testset_wav"
CHECKPOINT_PATH = r"best_mask_unet.pth"

# These MUST stay identical to the values used in src/enhance.py at inference.
SR = 16000
N_FFT = 512
HOP_LENGTH = 256

DURATION = 5.0
TARGET_LEN = int(SR * DURATION)
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 1e-5
PATIENCE = 10
USE_AMP = True

# Optional extra robustness: round-trip some training examples through
# low-bitrate MP3 so the model also learns to handle compression artifacts,
# not just acoustic noise. Set to 0.0 to disable. Requires ffmpeg on PATH.
MP3_AUGMENT_PROB = 0.0
# =============================================================================


def mp3_compress_augment(wav, sr=SR, bitrate_choices=("16k", "24k", "32k", "48k")):
    import subprocess
    import tempfile

    bitrate = random.choice(bitrate_choices)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "in.wav")
        mp3_path = os.path.join(tmp, "out.mp3")
        back_path = os.path.join(tmp, "back.wav")

        sf.write(wav_path, wav, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "quiet", "-i", wav_path, "-b:a", bitrate, mp3_path],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp3_path, "-ar", str(sr), "-ac", "1", back_path],
            check=True,
        )
        compressed, _ = librosa.load(back_path, sr=sr)

    return librosa.util.fix_length(compressed, size=len(wav)).astype(np.float32)


def build_paired_file_list(noisy_dir, clean_dir):
    """VoiceBank-DEMAND has identical filenames in the noisy/ and clean/
    folders (flat directory, e.g. p226_001.wav in both). Pair them by name."""
    noisy_files = {os.path.basename(f): f for f in glob.glob(os.path.join(noisy_dir, "*.wav"))}
    clean_files = {os.path.basename(f): f for f in glob.glob(os.path.join(clean_dir, "*.wav"))}

    common = sorted(set(noisy_files) & set(clean_files))
    missing = (set(noisy_files) | set(clean_files)) - set(common)
    if missing:
        print(f"Warning: {len(missing)} files present in only one of the two folders, skipping them")

    pairs = [(noisy_files[name], clean_files[name]) for name in common]
    assert len(pairs) > 0, f"No matching noisy/clean filename pairs found between {noisy_dir} and {clean_dir}"
    return pairs


class PairedSpeechDataset(Dataset):
    """Loads REAL recorded noisy/clean pairs -- no synthetic mixing needed,
    the dataset already gives you both sides."""

    def __init__(self, pairs, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 target_len=TARGET_LEN, mp3_augment_prob=0.0):
        self.pairs = pairs
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_len = target_len
        self.mp3_augment_prob = mp3_augment_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]

        noisy_wav, _ = librosa.load(noisy_path, sr=self.sr)
        clean_wav, _ = librosa.load(clean_path, sr=self.sr)

        # Random crop/pad both to a fixed length (keep them aligned -- same
        # crop offset for both, since they're time-synchronized recordings)
        if len(noisy_wav) > self.target_len:
            start = random.randint(0, len(noisy_wav) - self.target_len)
            noisy_wav = noisy_wav[start:start + self.target_len]
            clean_wav = clean_wav[start:start + self.target_len]
        else:
            noisy_wav = np.pad(noisy_wav, (0, self.target_len - len(noisy_wav)))
            clean_wav = np.pad(clean_wav, (0, self.target_len - len(clean_wav)))

        if self.mp3_augment_prob > 0 and random.random() < self.mp3_augment_prob:
            try:
                noisy_wav = mp3_compress_augment(noisy_wav, sr=self.sr)
            except Exception as e:
                print("mp3 augmentation skipped:", e)

        noisy_stft = librosa.stft(noisy_wav, n_fft=self.n_fft, hop_length=self.hop_length)
        clean_stft = librosa.stft(clean_wav, n_fft=self.n_fft, hop_length=self.hop_length)

        noisy_mag = np.abs(noisy_stft)
        clean_mag = np.abs(clean_stft)

        # Ratio mask target, computed on RAW linear magnitude
        mask = clean_mag / (noisy_mag + 1e-8)
        mask = np.clip(mask, 0.0, 1.0)

        # Model INPUT is log1p-compressed + normalized.
        # enhance.py at inference must do exactly this same transform.
        noisy_mag_log = np.log1p(noisy_mag)
        noisy_mag_norm = noisy_mag_log / (noisy_mag_log.max() + 1e-8)

        noisy_tensor = torch.tensor(noisy_mag_norm, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return noisy_tensor, mask_tensor, noisy_wav.astype(np.float32), clean_wav.astype(np.float32)


def compute_snr(clean, test):
    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    min_len = min(len(clean), len(test))
    clean = clean[:min_len]
    test = test[:min_len]
    noise = clean - test
    signal_power = np.sum(clean ** 2) + 1e-12
    noise_power = np.sum(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def evaluate_snr(clean, noisy, enhanced):
    snr_noisy = compute_snr(clean, noisy)
    snr_enh = compute_snr(clean, enhanced)
    return snr_noisy, snr_enh, snr_enh - snr_noisy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_pairs = build_paired_file_list(NOISY_TRAIN_DIR, CLEAN_TRAIN_DIR)
    val_pairs = build_paired_file_list(NOISY_TEST_DIR, CLEAN_TEST_DIR)
    print("Train pairs:", len(train_pairs))
    print("Val pairs:", len(val_pairs))

    train_dataset = PairedSpeechDataset(train_pairs, mp3_augment_prob=MP3_AUGMENT_PROB)
    val_dataset = PairedSpeechDataset(val_pairs, mp3_augment_prob=0.0)  # no augmentation on val

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Data ready | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = UNetMask().to(device)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_cnt = 0

    print("\nStarting training...\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for noisy, mask, _, _ in train_loader:
            noisy = noisy.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                pred = model(noisy)
                loss = 0.7 * mse(pred, mask) + 0.3 * l1(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for noisy, mask, _, _ in val_loader:
                noisy = noisy.to(device)
                mask = mask.to(device)
                with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                    pred = model(noisy)
                    val_loss = 0.7 * mse(pred, mask) + 0.3 * l1(pred, mask)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1:03}/{NUM_EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("Best model saved to", CHECKPOINT_PATH)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered")
                break

    print("\nTraining finished")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.grid()
    plt.title("Loss Curves")
    plt.savefig("loss_curve.png")
    print("Saved loss_curve.png")

    # Quick sanity check on one validation sample
    os.makedirs("samples", exist_ok=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    noisy_tensor, target_mask_tensor, mixed_wave, clean_wave = val_dataset[0]
    clean_wave = librosa.util.fix_length(clean_wave, size=len(mixed_wave))

    with torch.no_grad():
        pred_mask = model(noisy_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()

    noisy_stft = librosa.stft(mixed_wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)

    pred_mask = np.clip(pred_mask, 0.0, 1.0)
    enhanced_mag = pred_mask * noisy_mag
    enh_stft = enhanced_mag * np.exp(1j * noisy_phase)
    enh_audio = librosa.istft(enh_stft, hop_length=HOP_LENGTH, length=len(mixed_wave))

    snr_noisy, snr_enh, delta_snr = evaluate_snr(clean_wave, mixed_wave, enh_audio)
    print("Evaluation Metrics")
    print(f"Noisy SNR: {snr_noisy:.2f} dB")
    print(f"Enhanced SNR: {snr_enh:.2f} dB")
    print(f"Delta SNR (Gain): {delta_snr:.2f} dB")

    sf.write("samples/noisy.wav", mixed_wave.astype(np.float32), SR)
    sf.write("samples/enhanced.wav", enh_audio.astype(np.float32), SR)
    print("Wrote samples/noisy.wav and samples/enhanced.wav for a quick listen")


if __name__ == "__main__":
    main()