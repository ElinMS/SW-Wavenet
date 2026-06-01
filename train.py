"""
SW-WaveNet Training Pipeline
=============================
Trains the full SW-WaveNet model end-to-end:
    WavegramNet + WaveNet Encoder + Classifier

The model learns to classify Machine IDs (00, 02, 04, 06) using only
normal operating sounds from the DCASE 2020 fan dataset. During training,
WavegramNet learns to extract meaningful wavegrams from raw audio.

Usage:
    python train.py
"""

import os
import re
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import numpy as np

from wavegram_net import WavegramNet
from wavenet_model import SWWaveNetClassifier


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

DATASET_DIR   = r"c:\wavenet_data\dev_data_fan\fan\train"
NUM_FILES     = None         # Use all files (None = all)
SAMPLE_RATE   = 16000
N_MELS        = 128
N_FFT         = 1024
HOP_LENGTH    = 512

# Training hyperparameters
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 50
VAL_SPLIT     = 0.2          # 20% for validation
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Output
CHECKPOINT_DIR = r"c:\wavenet_project\checkpoints"


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class DCASEFanDataset(Dataset):
    """
    Dataset for DCASE 2020 fan training data.

    Loads audio files, extracts log-mel spectrograms, and parses Machine ID
    from the filename as the classification label.

    File naming: normal_id_XX_YYYYYYYY.wav  →  label = XX
    Machine IDs: 00, 02, 04, 06  →  mapped to class indices 0, 1, 2, 3

    Returns:
        spectrogram: [1, Time, Freq=128]   — log-mel spectrogram
        waveform:    [1, Samples]           — raw audio waveform
        label:       int                    — machine ID class index
    """

    def __init__(self, dataset_dir, num_files=250, sample_rate=16000,
                 n_mels=128, n_fft=1024, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Collect and sort audio files
        all_wavs = sorted(glob.glob(os.path.join(dataset_dir, "*.wav")))
        self.audio_files = all_wavs[:num_files] if num_files else all_wavs

        # Discover machine IDs and create label mapping
        all_ids = sorted(set(self._parse_machine_id(f) for f in self.audio_files))
        self.id_to_label = {mid: idx for idx, mid in enumerate(all_ids)}
        self.label_to_id = {idx: mid for mid, idx in self.id_to_label.items()}
        self.num_classes = len(all_ids)

        print(f"  Dataset: {len(self.audio_files)} files, "
              f"{self.num_classes} classes: {self.id_to_label}")

    @staticmethod
    def _parse_machine_id(filepath):
        """Extract machine ID from filename like 'normal_id_00_00000000.wav'."""
        basename = os.path.basename(filepath)
        match = re.search(r'_id_(\d+)_', basename)
        if match:
            return match.group(1)
        raise ValueError(f"Cannot parse machine ID from: {basename}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        # Load audio
        y, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Branch 1: Log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # [n_mels, time] → [1, time, n_mels]  (channel, time, freq)
        spectrogram = torch.tensor(log_mel_spec, dtype=torch.float32)\
                           .transpose(0, 1).unsqueeze(0)

        # Branch 2: Raw waveform
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # [1, samples]

        # Label: machine ID → class index
        machine_id = self._parse_machine_id(audio_path)
        label = self.id_to_label[machine_id]

        return spectrogram, waveform, label


def collate_fn(batch):
    """
    Custom collate to handle variable-length spectrograms and waveforms.

    Pads all samples in the batch to the maximum length along the time axis.
    """
    spectrograms, waveforms, labels = zip(*batch)

    # Pad spectrograms: [1, Time, 128] → pad Time dimension
    max_spec_time = max(s.shape[1] for s in spectrograms)
    padded_specs = []
    for s in spectrograms:
        pad_len = max_spec_time - s.shape[1]
        # Pad on the right side of time dimension
        padded = F.pad(s, (0, 0, 0, pad_len))  # (freq_left, freq_right, time_left, time_right)
        padded_specs.append(padded)
    spectrograms = torch.stack(padded_specs)  # [B, 1, T, 128]

    # Pad waveforms: [1, Samples] → pad Samples dimension
    max_wav_len = max(w.shape[1] for w in waveforms)
    padded_wavs = []
    for w in waveforms:
        pad_len = max_wav_len - w.shape[1]
        padded = F.pad(w, (0, pad_len))
        padded_wavs.append(padded)
    waveforms = torch.stack(padded_wavs)      # [B, 1, Samples]

    labels = torch.tensor(labels, dtype=torch.long)  # [B]

    return spectrograms, waveforms, labels


# ══════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (spectrogram, waveform, labels) in enumerate(dataloader):
        spectrogram = spectrogram.to(device)
        waveform = waveform.to(device)
        labels = labels.to(device)

        # Forward pass (labels needed for ArcFace angular margin)
        logits = model(spectrogram, waveform, labels=labels)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 5 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)}  "
                  f"loss={loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate, return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for spectrogram, waveform, labels in dataloader:
        spectrogram = spectrogram.to(device)
        waveform = waveform.to(device)
        labels = labels.to(device)

        logits = model(spectrogram, waveform)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    print("=" * 70)
    print("  SW-WaveNet — End-to-End Training")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    # ── 1. Dataset ────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    dataset = DCASEFanDataset(
        dataset_dir=DATASET_DIR,
        num_files=NUM_FILES,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # Split into train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"  Train: {train_size} samples  |  Val: {val_size} samples")
    print(f"  Batches per epoch: {len(train_loader)}")

    # ── 2. Model ──────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    wavegram_net = WavegramNet()

    model = SWWaveNetClassifier(
        wavegram_net=wavegram_net,
        num_classes=dataset.num_classes,
        spec_channels=N_MELS,
        wavegram_channels=512,
        layers=4,
        blocks=3,
        dilation_channels=512,
        residual_channels=512,
        skip_channels=512,
        repr_dim=128,
        kernel_size=2,
        arcface_scale=30.0,
        arcface_margin=0.7,
    )
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Classes              : {dataset.num_classes} "
          f"({dataset.label_to_id})")

    # ── 3. Optimizer & Loss ───────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler: cosine annealing (paper specification)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # ── 4. Training Loop ──────────────────────────────────────────────
    print("\n[3/4] Training...")
    print("-" * 70)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch:>2}/{NUM_EPOCHS}  "
              f"| train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"| val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"| lr={lr:.1e}  | {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'id_to_label': dataset.id_to_label,
                'label_to_id': dataset.label_to_id,
            }, ckpt_path)
            print(f"    * Saved best model (val_loss={val_loss:.4f})")

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'id_to_label': dataset.id_to_label,
        'label_to_id': dataset.label_to_id,
    }, final_path)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoints   : {CHECKPOINT_DIR}")
    print(f"    ├── best_model.pt")
    print(f"    └── final_model.pt")
    print("=" * 70)
    print("\n  Next step: run extract_features.py with --trained flag")
    print("  to generate meaningful wavegrams using the trained WavegramNet.")


if __name__ == "__main__":
    main()
