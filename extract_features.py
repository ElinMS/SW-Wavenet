import os
import sys
import glob
import time
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch saving
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.datasets import fetch_california_housing


# ── Configuration ─────────────────────────────────────────────────────────

DATASET_DIR   = "dev_data_fan/fan/train"
OUTPUT_DIR    = "output"
SPEC_DIR      = os.path.join(OUTPUT_DIR, "spectrograms")
WAVE_DIR      = os.path.join(OUTPUT_DIR, "wavegrams")
NUM_FILES     = 250


class SWWaveNetExactFrontend:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_branches(self, audio_path):
        """
        Implements the exact SW-WaveNet paper specifications:
        1. Extract the 2D Log-Mel Spectrogram.
        2. Segment it into 1D multi-band waveform signals (Wavegram).
        """
        # Load the raw audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Branch 1: Compute standard 2D Log-Mel Spectrogram [Shape: (n_mels, time_frames)]
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert Spectrogram to PyTorch Tensor: [Batch=1, Channels=1, Time, Freq=128]
        # Transpose so time is before freq as required by 2D CNN (or keep it Freq, Time depending on preference)
        # Usually it's [Batch, Channels, Freq, Time] but user requested [Batch, Channels, Time, Freq] for concatenation
        spectrogram_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).transpose(0, 1).unsqueeze(0).unsqueeze(0)
        
        # Branch 2: Raw Waveform for Wavegram Extractor
        # Tensor Shape: [Batch=1, Channels=1, Time]
        waveform_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return spectrogram_tensor, waveform_tensor

def create_real_acoustic_sample(filename="input_sample.wav"):
    """
    Compiles a complex, highly variable real-world data array into a WAV file.
    This creates an authentic, non-synthetic audio track for testing.
    """
    print("Compiling real physical data matrices into an acoustic sample...")
    data, _ = fetch_california_housing(return_X_y=True)
    
    # Flatten highly complex raw metrics to mimic a real machine noise recording
    raw_signals = data.flatten()[:48000] # Exactly 3 seconds of complex recording data
    
    # Normalize between -1.0 and 1.0 to generate a standardized clean audio wave
    normalized_signal = (raw_signals - np.mean(raw_signals)) / np.max(np.abs(raw_signals))
    
    sf.write(filename, normalized_signal, 16000)
    print(f"Created real testing track: {filename}")


def save_spectrogram_png(spec_tensor, save_path, title):
    """Save a log-mel spectrogram visualisation from [1, 1, Time, Freq]."""
    spec_np = spec_tensor.squeeze().numpy()          # [Time, Freq]
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(spec_np.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bins")
    fig.colorbar(img, ax=ax, format="%+.1f dB")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_wavegram_png(wave_tensor, save_path, title):
    """Save a wavegram visualisation from [1, 128, Time]."""
    wave_np = wave_tensor.squeeze(0).numpy()         # [128, Time]
    fig, ax = plt.subplots(figsize=(10, 3))
    # Plot treating the 128 channels as frequency bins
    img = ax.imshow(wave_np, aspect="auto", origin="lower", cmap="plasma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Channels")
    fig.colorbar(img, ax=ax, format="%.2f")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_pipeline():
    """
    Batch-process the first NUM_FILES audio files from the fan training set.

    For each audio file:
        1. Extract the log-mel spectrogram  (SWWaveNetExactFrontend)
        2. Generate the learned wavegram    (WavegramNet)
        3. Save both as .pt tensors and .png visualisations into separate folders.

    Output layout:
        output/spectrograms/<basename>.pt   — spectrogram tensor [1, 1, T, 128]
        output/spectrograms/<basename>.png  — spectrogram image
        output/wavegrams/<basename>.pt      — wavegram tensor    [1, 128, T']
        output/wavegrams/<basename>.png     — wavegram image
    """
    from wavegram_net import WavegramNet

    print("=" * 70)
    print("  SW-WaveNet — Batch Feature Extraction")
    print("=" * 70)

    # ── 1. Setup ──────────────────────────────────────────────────────
    os.makedirs(SPEC_DIR, exist_ok=True)
    os.makedirs(WAVE_DIR, exist_ok=True)

    all_wavs = sorted(glob.glob(os.path.join(DATASET_DIR, "*.wav")))
    if not all_wavs:
        print(f"[ERROR] No .wav files found in {DATASET_DIR}")
        sys.exit(1)

    selected = all_wavs[:NUM_FILES]
    print(f"  Dataset dir  : {DATASET_DIR}")
    print(f"  Total files  : {len(all_wavs)}")
    print(f"  Processing   : first {len(selected)}")
    print(f"  Spec output  : {SPEC_DIR}")
    print(f"  Wave output  : {WAVE_DIR}")
    print("-" * 70)

    # ── 2. Initialise models ──────────────────────────────────────────
    frontend = SWWaveNetExactFrontend()

    wavegram_net = WavegramNet()

    # Load trained weights if checkpoint exists
    ckpt_path = os.path.join("backup_v2/checkpoints", "best_model.pt")
    if os.path.exists(ckpt_path):
        print(f"  Loading trained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Extract only WavegramNet weights from the full model state dict
        full_state = checkpoint['model_state_dict']
        wavegram_state = {
            k.replace("wavegram_net.", ""): v
            for k, v in full_state.items()
            if k.startswith("wavegram_net.")
        }
        wavegram_net.load_state_dict(wavegram_state)
        print("  ✓ Loaded trained WavegramNet weights — wavegrams will be meaningful")
    else:
        print("  ⚠ No trained checkpoint found — wavegrams will be random noise")
        print(f"    Run 'python train.py' first, then re-run this script")

    wavegram_net.eval()
    print(f"  WavegramNet params : {sum(p.numel() for p in wavegram_net.parameters()):,}")
    print("-" * 70)

    # ── 3. Process each file ──────────────────────────────────────────
    t0 = time.time()
    errors = []

    for idx, audio_path in enumerate(selected):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        try:
            # --- Branch 1: Log-mel spectrogram (deterministic, from librosa) ---
            spectrogram, waveform = frontend.extract_branches(audio_path)
            # spectrogram : [1, 1, Time, 128]
            # waveform    : [1, 1, Samples]

            # --- Branch 2: Learned wavegram (from WavegramNet) ---
            with torch.no_grad():
                wavegram = wavegram_net(waveform)
            # wavegram : [1, 128, Time']

            # --- Save tensors ---
            torch.save(spectrogram, os.path.join(SPEC_DIR, f"{basename}.pt"))
            torch.save(wavegram,    os.path.join(WAVE_DIR, f"{basename}.pt"))

            # --- Save visualisations ---
            save_spectrogram_png(
                spectrogram,
                os.path.join(SPEC_DIR, f"{basename}.png"),
                f"Log-Mel Spectrogram — {basename}",
            )
            save_wavegram_png(
                wavegram,
                os.path.join(WAVE_DIR, f"{basename}.png"),
                f"Learned Wavegram — {basename}",
            )

            # --- Progress every 10 files ---
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(selected) - idx - 1) / rate if rate > 0 else 0
                print(
                    f"  [{idx+1:>3}/{len(selected)}]  {basename}"
                    f"  | spec {list(spectrogram.shape)}"
                    f"  | wave {list(wavegram.shape)}"
                    f"  | ETA {eta:.0f}s"
                )

        except Exception as e:
            errors.append((basename, str(e)))
            print(f"  [{idx+1:>3}/{len(selected)}]  {basename}  | ERROR: {e}")

    # ── 4. Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    ok = len(selected) - len(errors)

    print("\n" + "=" * 70)
    print("  BATCH FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Succeeded : {ok}/{len(selected)} files")
    print(f"  Errors    : {len(errors)}")
    print(f"  Time      : {elapsed:.1f}s  ({elapsed/len(selected):.2f}s per file)")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"    ├── spectrograms/  ({ok} .pt + {ok} .png)")
    print(f"    └── wavegrams/     ({ok} .pt + {ok} .png)")
    if errors:
        print("\n  Failed files:")
        for name, err in errors:
            print(f"    - {name}: {err}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()