import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.enhance import enhance_audio

SR = 16000


# ---------------------------------------------------
# SNR
# ---------------------------------------------------
from scipy.signal import correlate

def compute_snr(clean, test):

    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    # -------- Align signals using cross-correlation --------
    corr = correlate(test, clean, mode="full")
    lag = np.argmax(corr) - (len(clean) - 1)

    if lag > 0:
        test = test[lag:]
    elif lag < 0:
        clean = clean[-lag:]

    min_len = min(len(clean), len(test))
    clean = clean[:min_len]
    test = test[:min_len]

    noise = clean - test

    signal_power = np.sum(clean ** 2) + 1e-12
    noise_power = np.sum(noise ** 2) + 1e-12

    return 10 * np.log10(signal_power / noise_power)


# ---------------------------------------------------
# Spectrogram Plot
# ---------------------------------------------------
def save_comparison_spectrogram(clean, noisy, enhanced):

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    audios = [clean, noisy, enhanced]

    titles = [
        "Clean Speech",
        "Noisy Speech",
        "Enhanced Speech"
    ]

    for ax, audio, title in zip(axes, audios, titles):

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio)),
            ref=np.max
        )

        img = librosa.display.specshow(
            D,
            sr=SR,
            hop_length=256,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=ax
        )

        ax.set_title(title, fontsize=13,fontweight="bold")

        if ax != axes[-1]:
            ax.set_xlabel("")

    fig.colorbar(img, ax=axes, format="%+2.0f dB")

    plt.suptitle(
        "Speech Enhancement Spectrogram Comparison",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    plt.savefig(
        "spectrogram_comparison.png",
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
def evaluate(clean_path, noisy_path):

    print("\nEnhancing noisy audio...\n")

    enhance_audio(noisy_path)

    clean, _ = librosa.load(clean_path, sr=SR)
    noisy, _ = librosa.load(noisy_path, sr=SR)

    # Load RAW enhanced signal for evaluation
    enhanced, _ = librosa.load("enhanced_raw.wav", sr=SR)

    snr_before = compute_snr(clean, noisy)
    snr_after = compute_snr(clean, enhanced)

    print("="*55)
    print("Speech Enhancement Evaluation")
    print("="*55)

    print(f"SNR Before Enhancement : {snr_before:.2f} dB")
    print(f"SNR After Enhancement  : {snr_after:.2f} dB")
    print(f"SNR Improvement        : {snr_after-snr_before:.2f} dB")

    print("="*55)

    print("\nSaving spectrograms...")

    save_comparison_spectrogram(
        clean,
        noisy,
        enhanced
    )
    print("Done!")
    print("\nGenerated:")
    print("spectrogram_comparison.png")


if __name__ == "__main__":

    clean_path = input("Enter clean audio path : ").strip()
    noisy_path = input("Enter noisy audio path : ").strip()

    evaluate(clean_path, noisy_path)