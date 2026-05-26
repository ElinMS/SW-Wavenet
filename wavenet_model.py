"""
WaveNet Backbone — Dilated causal convolution encoder for generating
representation vectors from 2D time-frequency features.

Reference: vincentherrmann/pytorch-wavenet (adapted for encoding, not generation)

In the SW-WaveNet architecture, WaveNet is used as a feature encoder (not a
generative auto-regressive model). Both the log-mel spectrogram and the learned
wavegram are reshaped from 2D (Time × Freq) into 1D sequences where frequency
bins become input channels, then processed through stacked dilated convolution
blocks to produce fixed-size representation vectors.

Architecture (per branch):
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. Start Conv (1×1)                                           │
    │     Project input channels → residual_channels                 │
    │                                                                │
    │  2. Stacked Dilated Blocks (layers × blocks):                  │
    │     ┌────────────────────────────────────────────────────────┐  │
    │     │  Dilated Conv (filter) → tanh  ─┐                     │  │
    │     │                                 ├→ element-wise mult  │  │
    │     │  Dilated Conv (gate)  → sigmoid ┘     (gated act.)    │  │
    │     │       │                                               │  │
    │     │       ├─→ 1×1 Conv → + input  (residual connection)   │  │
    │     │       └─→ 1×1 Conv → accumulate (skip connection)     │  │
    │     └────────────────────────────────────────────────────────┘  │
    │     Dilations: 1, 2, 4, 8, 16, 32, ... (per block)            │
    │                                                                │
    │  3. End Processing:                                            │
    │     Sum skip connections → ReLU → 1×1 Conv → ReLU → 1×1 Conv  │
    │                                                                │
    │  4. Global Average Pooling → Representation Vector             │
    └─────────────────────────────────────────────────────────────────┘

Dual-Branch Encoder:
    Raw Waveform ─┬─ Feature Extraction ──→ Spectrogram ──→ WaveNet ──→ Repr Vec
                  └─ WavegramNet ─────────→ Wavegram ─────→ WaveNet ──→ Repr Vec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class WaveNetLayer(nn.Module):
    """
    A single WaveNet dilated convolution layer with gated activation unit.

    Implements:
        filter  = tanh( DilatedConv(x) )
        gate    = σ( DilatedConv(x) )
        z       = filter ⊙ gate              (gated activation)
        skip    = Conv1×1(z)                  (skip connection output)
        residual= Conv1×1(z) + x              (residual connection)

    Args:
        residual_channels:  Width of the residual path
        dilation_channels:  Width inside the gated activation
        skip_channels:      Width of the skip-connection output
        kernel_size:        Convolution kernel size (default 2)
        dilation:           Dilation factor for this layer
        bias:               Whether conv layers use bias terms
    """

    def __init__(self, residual_channels, dilation_channels, skip_channels,
                 kernel_size=2, dilation=1, bias=True):
        super(WaveNetLayer, self).__init__()

        self.dilation = dilation
        self.kernel_size = kernel_size

        # Dilated convolutions for the gated activation unit
        # Padding = dilation * (kernel_size - 1) gives causal (left) padding
        pad = dilation * (kernel_size - 1)
        self.filter_conv = nn.Conv1d(
            residual_channels, dilation_channels,
            kernel_size=kernel_size, dilation=dilation,
            padding=pad, bias=bias
        )
        self.gate_conv = nn.Conv1d(
            residual_channels, dilation_channels,
            kernel_size=kernel_size, dilation=dilation,
            padding=pad, bias=bias
        )

        # 1×1 convolutions for residual and skip projections
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1, bias=bias)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [Batch, residual_channels, Time]

        Returns:
            residual: [Batch, residual_channels, Time]  (same length as input)
            skip:     [Batch, skip_channels, Time]
        """
        # Dilated convolutions with causal padding
        f = self.filter_conv(x)
        g = self.gate_conv(x)

        # Trim extra padding to maintain causal property and match input length
        if f.size(-1) != x.size(-1):
            f = f[..., :x.size(-1)]
            g = g[..., :x.size(-1)]

        # Gated activation unit
        z = torch.tanh(f) * torch.sigmoid(g)

        # Skip connection output
        skip = self.skip_conv(z)

        # Residual connection
        residual = self.residual_conv(z) + x

        return residual, skip


# ---------------------------------------------------------------------------
# WaveNet Backbone (single branch)
# ---------------------------------------------------------------------------

class WaveNetBackbone(nn.Module):
    """
    WaveNet encoder backbone that produces a representation vector
    from a 1D multi-channel sequence.

    The backbone stacks multiple dilated convolution layers arranged in
    blocks.  Within each block the dilation doubles per layer
    (1, 2, 4, 8, …), giving the network an exponentially large receptive
    field.  Skip connections from every layer are summed and passed through
    two 1×1 convolutions before global average pooling collapses the time
    axis into a fixed-size representation vector.

    Input:  [Batch, in_channels, Time]
    Output: [Batch, repr_dim]

    Args:
        in_channels:        Number of input channels (freq bins or C×F)
        layers:             Number of dilated layers per block
        blocks:             Number of repeated blocks
        dilation_channels:  Channels inside gated activation
        residual_channels:  Channels on the residual path
        skip_channels:      Channels for skip connections
        end_channels:       Intermediate channels in end processing
        repr_dim:           Dimensionality of the output representation vector
        kernel_size:        Kernel size for dilated convolutions
        bias:               Whether to use bias in conv layers
    """

    def __init__(self,
                 in_channels,
                 layers=6,
                 blocks=2,
                 dilation_channels=32,
                 residual_channels=64,
                 skip_channels=128,
                 end_channels=128,
                 repr_dim=128,
                 kernel_size=2,
                 bias=True):
        super(WaveNetBackbone, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.repr_dim = repr_dim

        # --- Start: project input channels to residual width ---
        self.start_conv = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        self.start_bn = nn.BatchNorm1d(residual_channels)

        # --- Core: stacked dilated layers ---
        self.wavenet_layers = nn.ModuleList()
        for b in range(blocks):
            for i in range(layers):
                dilation = 2 ** i
                self.wavenet_layers.append(
                    WaveNetLayer(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        bias=bias,
                    )
                )

        # --- End: process accumulated skip connections ---
        self.end_conv1 = nn.Conv1d(skip_channels, end_channels, 1, bias=bias)
        self.end_bn1 = nn.BatchNorm1d(end_channels)
        self.end_conv2 = nn.Conv1d(end_channels, repr_dim, 1, bias=bias)
        self.end_bn2 = nn.BatchNorm1d(repr_dim)

        # Receptive field (for information / debugging)
        self.receptive_field = self._calc_receptive_field()

    # ---- helpers ----

    def _calc_receptive_field(self):
        """Total receptive field in samples."""
        rf = 1
        for _ in range(self.blocks):
            for i in range(self.layers):
                rf += (self.kernel_size - 1) * (2 ** i)
        return rf

    # ---- forward ----

    def forward(self, x):
        """
        Args:
            x: [Batch, in_channels, Time]

        Returns:
            repr_vec: [Batch, repr_dim]
        """
        # Project to residual channels
        x = F.relu(self.start_bn(self.start_conv(x)))

        # Accumulate skip connections across all layers
        skip_sum = 0
        for layer in self.wavenet_layers:
            x, skip = layer(x)
            skip_sum = skip_sum + skip

        # End processing on accumulated skips
        out = F.relu(skip_sum)
        out = F.relu(self.end_bn1(self.end_conv1(out)))
        out = self.end_bn2(self.end_conv2(out))

        # Global Average Pooling → fixed-size representation vector
        repr_vec = out.mean(dim=-1)  # [Batch, repr_dim]

        return repr_vec


# ---------------------------------------------------------------------------
# Dual-Branch SW-WaveNet Encoder
# ---------------------------------------------------------------------------

class SWWaveNetEncoder(nn.Module):
    """
    Complete SW-WaveNet Dual-Branch Encoder.

    Processes both the log-mel spectrogram and the learned wavegram through
    separate WaveNet backbones to produce representation vectors.

    Input shapes (from the frontend):
        spectrogram : [Batch, 1, Time,  Freq=128]   (log-mel spectrogram)
        wavegram    : [Batch, 4, Time', Freq=128]   (learned wavegram)

    The 2D features are reshaped into 1D sequences by treating frequency
    bins (and wavegram output channels) as input channels:
        spectrogram → [Batch, 128, Time]   (128 freq-bin channels)
        wavegram    → [Batch, 512, Time']  (4×128 = 512 channels)

    Output:
        spec_repr : [Batch, repr_dim]   representation vector from spectrogram
        wave_repr : [Batch, repr_dim]   representation vector from wavegram

    Args:
        spec_channels:      Input channels for spectrogram branch (n_mels)
        wavegram_channels:  Input channels for wavegram branch (C × F)
        layers:             Dilated layers per block in each WaveNet
        blocks:             Number of dilation blocks in each WaveNet
        dilation_channels:  Channels inside gated activation
        residual_channels:  Channels on residual path
        skip_channels:      Channels for skip connections
        end_channels:       Intermediate channels in end processing
        repr_dim:           Size of the output representation vector
        kernel_size:        Kernel size for dilated convolutions
    """

    def __init__(self,
                 spec_channels=128,
                 wavegram_channels=512,
                 layers=6,
                 blocks=2,
                 dilation_channels=32,
                 residual_channels=64,
                 skip_channels=128,
                 end_channels=128,
                 repr_dim=128,
                 kernel_size=2):
        super(SWWaveNetEncoder, self).__init__()

        self.repr_dim = repr_dim

        # WaveNet backbone for the spectrogram branch
        self.wavenet_spec = WaveNetBackbone(
            in_channels=spec_channels,
            layers=layers,
            blocks=blocks,
            dilation_channels=dilation_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            repr_dim=repr_dim,
            kernel_size=kernel_size,
        )

        # WaveNet backbone for the wavegram branch
        self.wavenet_wave = WaveNetBackbone(
            in_channels=wavegram_channels,
            layers=layers,
            blocks=blocks,
            dilation_channels=dilation_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            repr_dim=repr_dim,
            kernel_size=kernel_size,
        )

    # ---- input reshaping ----

    @staticmethod
    def _reshape_spectrogram(spec):
        """Reshape [Batch, 1, Time, Freq=128] → [Batch, 128, Time]."""
        # Squeeze the single input-channel dim, then swap Time ↔ Freq
        return spec.squeeze(1).transpose(1, 2)

    @staticmethod
    def _reshape_wavegram(wavegram):
        """Reshape [Batch, C=4, Time, Freq=128] → [Batch, C×Freq=512, Time]."""
        B, C, T, F = wavegram.shape
        # (B, C, T, F) → (B, C, F, T) → (B, C*F, T)
        return wavegram.permute(0, 1, 3, 2).reshape(B, C * F, T)

    # ---- forward ----

    def forward(self, spectrogram, wavegram):
        """
        Args:
            spectrogram: [Batch, 1, Time,  Freq=128]
            wavegram:    [Batch, 4, Time', Freq=128]

        Returns:
            spec_repr: [Batch, repr_dim]  — representation from spectrogram branch
            wave_repr: [Batch, repr_dim]  — representation from wavegram branch
        """
        # Reshape 2D features → 1D multi-channel sequences
        spec_1d = self._reshape_spectrogram(spectrogram)   # [B, 128, T]
        wave_1d = self._reshape_wavegram(wavegram)         # [B, 512, T']

        # Pass through respective WaveNet backbones
        spec_repr = self.wavenet_spec(spec_1d)
        wave_repr = self.wavenet_wave(wave_1d)

        return spec_repr, wave_repr


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  WaveNet Backbone — Standalone Architecture Test")
    print("=" * 70)

    # ---- Test individual backbone ----
    print("\n--- Single WaveNetBackbone (spectrogram-like input) ---")
    backbone = WaveNetBackbone(in_channels=128, layers=6, blocks=2, repr_dim=128)
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"  Receptive field   : {backbone.receptive_field} samples")
    print(f"  Total parameters  : {total_params:,}")

    x = torch.randn(1, 128, 50)  # [Batch, Freq=128, Time=50]
    with torch.no_grad():
        out = backbone(x)
    print(f"  Input shape       : {list(x.shape)}")
    print(f"  Output shape      : {list(out.shape)}")

    # ---- Test dual-branch encoder ----
    print("\n--- SWWaveNetEncoder (dual-branch) ---")
    encoder = SWWaveNetEncoder(
        spec_channels=128, wavegram_channels=512,
        layers=6, blocks=2, repr_dim=128
    )
    total_params_enc = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters  : {total_params_enc:,}")

    spec = torch.randn(1, 1, 50, 128)   # [B, 1, T, F=128]
    wave = torch.randn(1, 4, 50, 128)   # [B, 4, T', F=128]
    print(f"  Spectrogram input : {list(spec.shape)}")
    print(f"  Wavegram input    : {list(wave.shape)}")

    with torch.no_grad():
        spec_repr, wave_repr = encoder(spec, wave)

    print(f"  Spec repr output  : {list(spec_repr.shape)}")
    print(f"  Wave repr output  : {list(wave_repr.shape)}")

    print("\n" + "=" * 70)
    print("  Spectrogram WaveNet receptive field :",
          encoder.wavenet_spec.receptive_field, "frames")
    print("  Wavegram   WaveNet receptive field :",
          encoder.wavenet_wave.receptive_field, "frames")
    print("=" * 70)
    print("\n✓ SW-WaveNet dual-branch encoder verified successfully!")
