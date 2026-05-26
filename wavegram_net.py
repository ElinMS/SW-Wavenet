"""
WavegramNet — A 1D CNN that learns a time-frequency representation (Wavegram)
directly from the raw audio waveform.

Reference:
    "A Wavegram is a time-frequency representation similar to the log-mel
    spectrogram, but it is obtained by a neural network learning from the
    raw waveform." — SW-WaveNet paper

Architecture (layer-by-layer):
    Stage | Operation      | Parameters              | Output Effect
    ------|----------------|-------------------------|-------------------------------
    1     | Conv1D         | kernel=11, stride=5     | Initial temporal feature extraction
    2     | Conv1D Block   | Conv1D + BN + ReLU      | Learns low-level waveform patterns
    3     | MaxPooling1D   | stride=4                | Downsampling
    4     | Conv1D Block   | Conv1D + BN + ReLU      | Mid-level features
    5     | MaxPooling1D   | stride=4                | Further downsampling
    6     | Conv1D Block   | Conv1D + BN + ReLU      | Higher-level temporal features
    7     | MaxPooling1D   | stride=4                | Compresses temporal dimension
    8     | Reshape        | reshape to (N, C, T, F) | Converts 1D features into 2D Wavegram
    9     | Output         | Wavegram                | Used by CNN/WaveNet backend

Channel progression: 1 -> 64 -> 128 -> 256 -> 512
Reshape logic: 512 channels -> C=4, F=128 (matches n_mels for concatenation)
"""

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """A single Conv1D + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class WavegramNet(nn.Module):
    """
    Extracts a learned 2D Wavegram from a raw 1D audio waveform.

    Input:  [Batch, 1, Time]           — raw waveform (e.g. 16 000 samples for 1 s @ 16 kHz)
    Output: [Batch, C, T', F]          — learned time-frequency Wavegram

    With default settings (n_freq=128, final_channels=512):
        C = final_channels // n_freq = 512 // 128 = 4
        F = n_freq = 128
    """

    def __init__(self, n_freq=128, final_channels=512):
        super(WavegramNet, self).__init__()

        self.n_freq = n_freq
        self.final_channels = final_channels
        self.reshape_channels = final_channels // n_freq  # C = 4

        # Stage 1: Initial Conv1D — large kernel to capture raw waveform patterns
        self.initial_conv = nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=5)
        self.initial_bn = nn.BatchNorm1d(64)
        self.initial_relu = nn.ReLU()

        # Stage 2-3: Conv1D Block 1 + MaxPool
        self.block1 = Conv1DBlock(64, 128)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Stage 4-5: Conv1D Block 2 + MaxPool
        self.block2 = Conv1DBlock(128, 256)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Stage 6-7: Conv1D Block 3 + MaxPool
        self.block3 = Conv1DBlock(256, final_channels)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        """
        Args:
            x: Raw waveform tensor of shape [Batch, 1, Time]

        Returns:
            wavegram: Learned 2D representation of shape [Batch, C, T', F]
                      where C = final_channels // n_freq, F = n_freq
        """
        # Stage 1: Initial temporal feature extraction
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))

        # Stages 2-7: Progressive feature learning + downsampling
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        # x shape: [Batch, 512, T']

        # Stage 8: Reshape 1D features into 2D Wavegram
        # Step 1 — swap Time and Channel axes: [Batch, T', 512]
        x = x.transpose(1, 2)

        # Step 2 — split channel dim into (C, F): [Batch, T', C=4, F=128]
        batch_size, time_steps, _ = x.shape
        x = x.view(batch_size, time_steps, self.reshape_channels, self.n_freq)

        # Step 3 — rearrange to target shape: [Batch, C=4, T', F=128]
        wavegram = x.permute(0, 2, 1, 3)

        return wavegram


if __name__ == "__main__":
    print("=" * 60)
    print("  WavegramNet — Standalone Architecture Test")
    print("=" * 60)

    # Simulate 1 second of audio at 16 kHz
    waveform = torch.randn(1, 1, 16000)
    print(f"\nInput waveform shape : {list(waveform.shape)}")

    model = WavegramNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters     : {total_params:,}")

    with torch.no_grad():
        wavegram = model(waveform)

    print(f"Output Wavegram shape: {list(wavegram.shape)}")
    print(f"  -> [Batch={wavegram.shape[0]}, C={wavegram.shape[1]}, "
          f"T={wavegram.shape[2]}, F={wavegram.shape[3]}]")

    print("\nWavegramNet architecture verified successfully!")
