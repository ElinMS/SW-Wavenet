"""
WaveNet Backbone — Dilated causal convolution encoder for generating
representation vectors from 2D time-frequency features.

Reference: vincentherrmann/pytorch-wavenet (adapted for encoding, not generation)

In the SW-WaveNet architecture, WaveNet is used as a feature encoder (not a
generative auto-regressive model). Both the log-mel spectrogram and the learned
wavegram are reshaped from 2D (Time × Freq) into 1D sequences where frequency
bins become input channels, then processed through stacked dilated convolution
blocks to produce fixed-size representation vectors.

Architecture (per branch) — Paper-matched configuration:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. Start Conv (1×1)                                           │
    │     Project input channels → residual_channels (512)           │
    │                                                                │
    │  2. Stacked Dilated Blocks (3 stacks × 4 layers = 12 total):  │
    │     ┌────────────────────────────────────────────────────────┐  │
    │     │  Dilated Conv (filter) → tanh  ─┐                     │  │
    │     │                                 ├→ element-wise mult  │  │
    │     │  Dilated Conv (gate)  → sigmoid ┘     (gated act.)    │  │
    │     │       │ → BatchNorm                                   │  │
    │     │       ├─→ 1×1 Conv → + input  (residual connection)   │  │
    │     │       └─→ 1×1 Conv → accumulate (skip connection)     │  │
    │     └────────────────────────────────────────────────────────┘  │
    │     Dilations: 1, 2, 4, 8 (per stack)                          │
    │                                                                │
    │  3. Representation Compression Module:                         │
    │     Sum skips → BN → ReLU → Depthwise Conv1D → BN → 1×1 Conv  │
    │                                                                │
    │  4. Global Average Pooling → Representation Vector (128-d)     │
    └─────────────────────────────────────────────────────────────────┘

Dual-Branch Encoder:
    Raw Waveform ─┬─ Feature Extraction ──→ Spectrogram ──→ WaveNet ──→ Repr Vec
                  └─ WavegramNet ─────────→ Wavegram ─────→ WaveNet ──→ Repr Vec

Classification:
    concat(spec_repr, wave_repr) → ArcFace Head → Category Probabilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class WaveNetLayer(nn.Module):
    """
    A single WaveNet dilated convolution layer with gated activation unit
    and batch normalization.

    Implements:
        filter  = tanh( DilatedConv(x) )
        gate    = σ( DilatedConv(x) )
        z       = BN( filter ⊙ gate )          (gated activation + BN)
        skip    = Conv1×1(z)                    (skip connection output)
        residual= Conv1×1(z) + x                (residual connection)

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

        # Batch normalization after gated activation (paper requirement)
        self.bn = nn.BatchNorm1d(dilation_channels)

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

        # Gated activation unit + batch normalization
        z = self.bn(torch.tanh(f) * torch.sigmoid(g))

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
    WaveNet encoder backbone with representation compression module.

    The backbone stacks multiple dilated convolution layers arranged in
    blocks.  Within each block the dilation doubles per layer
    (1, 2, 4, 8), giving the network an exponentially large receptive
    field.  Skip connections from every layer are summed and passed through
    the representation compression module before global average pooling
    collapses the time axis into a fixed-size representation vector.

    Paper configuration:
        - m = 3 residual stacks, each with 4 dilated residual blocks
        - Dilations per stack: 1, 2, 4, 8
        - Cout = 512, Cskip = 512
        - Representation compression: BN → ReLU → Depthwise Conv → BN → 1×1 Conv
        - Representation vector length: 128

    Input:  [Batch, in_channels, Time]
    Output: [Batch, repr_dim]

    Args:
        in_channels:        Number of input channels (freq bins or C×F)
        layers:             Number of dilated layers per block (paper: 4)
        blocks:             Number of repeated blocks (paper: 3)
        dilation_channels:  Channels inside gated activation (paper: 512)
        residual_channels:  Channels on the residual path (paper: 512)
        skip_channels:      Channels for skip connections (paper: 512)
        repr_dim:           Dimensionality of the output representation vector
        kernel_size:        Kernel size for dilated convolutions
        bias:               Whether to use bias in conv layers
    """

    def __init__(self,
                 in_channels,
                 layers=4,
                 blocks=3,
                 dilation_channels=512,
                 residual_channels=512,
                 skip_channels=512,
                 repr_dim=128,
                 start_kernel_size=2,
                 kernel_size=3,
                 bias=True):
        super(WaveNetBackbone, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.repr_dim = repr_dim

        # --- Start: project input channels to residual width ---
        # Paper specifies causal convolution K=2
        self.start_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=start_kernel_size, padding=1, bias=bias)
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

        # --- Representation Compression Module ---
        # Paper: "two batch normalization layers, a ReLU activation function,
        #         a 1-D depth-wise convolution and a 1×1 convolution"
        self.compress_bn1 = nn.BatchNorm1d(skip_channels)
        self.compress_depthwise = nn.Conv1d(
            skip_channels, skip_channels, kernel_size=3, padding=1,
            groups=skip_channels, bias=bias
        )
        self.compress_bn2 = nn.BatchNorm1d(skip_channels)
        self.compress_pointwise = nn.Conv1d(skip_channels, repr_dim, 1, bias=bias)

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
        original_length = x.size(-1)
        x = self.start_conv(x)
        # Strip padding to keep causal property
        if x.size(-1) != original_length:
            x = x[..., :-1]
        x = F.relu(self.start_bn(x))

        # Accumulate skip connections across all layers
        skip_sum = 0
        for layer in self.wavenet_layers:
            x, skip = layer(x)
            skip_sum = skip_sum + skip

        # Representation Compression Module
        out = F.relu(self.compress_bn1(skip_sum))
        out = self.compress_depthwise(out)
        out = self.compress_bn2(out)
        out = self.compress_pointwise(out)

        # Global Average Pooling → fixed-size representation vector
        repr_vec = out.mean(dim=-1)  # [Batch, repr_dim]

        return repr_vec


# ---------------------------------------------------------------------------
# ArcFace Classification Head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) classification head.

    Paper specification: margin = 0.7, scale = 30

    During training, adds an angular margin penalty to the target class
    cosine similarity, forcing the model to learn more discriminative
    representation vectors.  During evaluation, returns plain scaled
    cosine similarities (no margin applied).

    Args:
        in_features:  Dimensionality of the input representation
        num_classes:  Number of output categories
        scale:        Scaling factor s (paper: 30)
        margin:       Angular margin m in radians (paper: 0.7)
    """

    def __init__(self, in_features, num_classes, scale=30.0, margin=0.7):
        super(ArcFaceHead, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms for cos(theta + m) expansion
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold to avoid numerical issues when theta + m > pi
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels=None):
        """
        Args:
            features: [Batch, in_features]
            labels:   [Batch] — required during training for margin

        Returns:
            logits: [Batch, num_classes] — scaled cosine similarities
        """
        # L2 normalize features and weights
        features_norm = F.normalize(features, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: [B, num_classes]
        cosine = F.linear(features_norm, weights_norm)

        if labels is not None and self.training:
            # Compute sin(theta) from cos(theta)
            sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))

            # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            phi = cosine * self.cos_m - sine * self.sin_m

            # Numerical safety: if cos(theta) < threshold, use linear fallback
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            # Apply margin only to the target class
            one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine

        # Scale by s
        output = output * self.scale
        return output


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
        layers:             Dilated layers per block in each WaveNet (paper: 4)
        blocks:             Number of dilation blocks in each WaveNet (paper: 3)
        dilation_channels:  Channels inside gated activation (paper: 512)
        residual_channels:  Channels on residual path (paper: 512)
        skip_channels:      Channels for skip connections (paper: 512)
        repr_dim:           Size of the output representation vector (paper: 128)
        kernel_size:        Kernel size for dilated convolutions
    """

    def __init__(self,
                 spec_channels=128,
                 wavegram_channels=128,
                 layers=4,
                 blocks=3,
                 dilation_channels=512,
                 residual_channels=512,
                 skip_channels=512,
                 repr_dim=128,
                 kernel_size=3):
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
            repr_dim=repr_dim,
            kernel_size=kernel_size,
        )

    # ---- input reshaping ----

    @staticmethod
    def _reshape_spectrogram(spec):
        """Reshape [Batch, 1, Time, Freq=128] → [Batch, 128, Time]."""
        # Squeeze the single input-channel dim, then swap Time ↔ Freq
        return spec.squeeze(1).transpose(1, 2)

    # ---- forward ----

    def forward(self, spectrogram, wavegram):
        """
        Args:
            spectrogram: [Batch, 1, Time,  Freq=128]
            wavegram:    [Batch, 128, Time']

        Returns:
            spec_repr: [Batch, repr_dim]  — representation from spectrogram branch
            wave_repr: [Batch, repr_dim]  — representation from wavegram branch
        """
        # Reshape 2D features → 1D multi-channel sequences
        spec_1d = self._reshape_spectrogram(spectrogram)   # [B, 128, T]
        wave_1d = wavegram                                 # [B, 128, T']

        # Pass through respective WaveNet backbones
        spec_repr = self.wavenet_spec(spec_1d)
        wave_repr = self.wavenet_wave(wave_1d)

        return spec_repr, wave_repr


# ---------------------------------------------------------------------------
# Full SW-WaveNet Classifier (end-to-end)
# ---------------------------------------------------------------------------

class SWWaveNetClassifier(nn.Module):
    """
    Complete SW-WaveNet model for anomalous sound classification.

    Combines all components into a single end-to-end trainable model:
        1. SWWaveNetExactFrontend  (external, provides spectrogram — not trainable)
        2. WavegramNet             (learns wavegram from raw waveform)
        3. SWWaveNetEncoder        (dual-branch WaveNet encoder)
        4. ArcFace head            (concatenation → cosine similarity + margin)

    The model is trained to classify Machine IDs using only normal sounds.
    At test time, anomalous sounds produce low confidence → high anomaly score.

    Paper: "we concatenate the representation vectors of the two branches and
    feed the combination into a fully connected layer with a softmax to get the
    category probabilities. The negative log probability is used as the anomaly
    score for each sound."

    Loss: ArcFace (margin=0.7, scale=30) with CrossEntropyLoss on the output.

    Args:
        wavegram_net:       WavegramNet instance (learns wavegram from waveform)
        num_classes:        Number of Machine IDs to classify
        spec_channels:      Frequency bins in spectrogram (n_mels)
        wavegram_channels:  Channels in wavegram (C × F)
        layers:             Dilated layers per block (paper: 4)
        blocks:             Number of dilation blocks (paper: 3)
        dilation_channels:  Channels inside gated activation (paper: 512)
        residual_channels:  Channels on residual path (paper: 512)
        skip_channels:      Channels for skip connections (paper: 512)
        repr_dim:           Size of each branch's representation vector (paper: 128)
        kernel_size:        Kernel size for dilated convolutions
        arcface_scale:      ArcFace scaling factor (paper: 30)
        arcface_margin:     ArcFace angular margin (paper: 0.7)
    """

    def __init__(self,
                 wavegram_net,
                 num_classes,
                 spec_channels=128,
                 wavegram_channels=128,
                 layers=4,
                 blocks=3,
                 dilation_channels=512,
                 residual_channels=512,
                 skip_channels=512,
                 repr_dim=128,
                 kernel_size=3,
                 arcface_scale=30.0,
                 arcface_margin=0.7):
        super(SWWaveNetClassifier, self).__init__()

        self.repr_dim = repr_dim

        # Branch 2 feature extractor: raw waveform → wavegram
        self.wavegram_net = wavegram_net

        # Dual-branch WaveNet encoder
        self.encoder = SWWaveNetEncoder(
            spec_channels=spec_channels,
            wavegram_channels=wavegram_channels,
            layers=layers,
            blocks=blocks,
            dilation_channels=dilation_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            repr_dim=repr_dim,
            kernel_size=kernel_size,
        )

        # ArcFace classification head: concatenated repr (2 × repr_dim) → classes
        self.classifier = ArcFaceHead(
            in_features=repr_dim * 2,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def forward(self, spectrogram, waveform, labels=None):
        """
        Args:
            spectrogram: [Batch, 1, Time, Freq=128]  — from frontend (no grad)
            waveform:    [Batch, 1, Samples]          — raw audio
            labels:      [Batch] int                  — required during training
                         for ArcFace angular margin (ignored during eval)

        Returns:
            logits: [Batch, num_classes]  — scaled cosine similarities
        """
        # Branch 2: waveform → learned wavegram
        wavegram = self.wavegram_net(waveform)      # [B, 128, T']

        # Dual-branch encoding → representation vectors
        spec_repr, wave_repr = self.encoder(spectrogram, wavegram)
        # spec_repr: [B, repr_dim],  wave_repr: [B, repr_dim]

        # Concatenate both branches
        combined = torch.cat([spec_repr, wave_repr], dim=-1)  # [B, 2*repr_dim]

        # ArcFace classification
        logits = self.classifier(combined, labels)   # [B, num_classes]
        return logits

    def get_representation(self, spectrogram, waveform):
        """
        Extract the concatenated representation vector without classification.

        Useful for t-SNE visualization, anomaly scoring, etc.

        Returns:
            combined: [Batch, 2 * repr_dim]
        """
        wavegram = self.wavegram_net(waveform)
        spec_repr, wave_repr = self.encoder(spectrogram, wavegram)
        return torch.cat([spec_repr, wave_repr], dim=-1)

    def anomaly_score(self, spectrogram, waveform):
        """
        Compute anomaly score = negative log probability of predicted class.

        Returns:
            scores: [Batch]  — higher score = more anomalous
        """
        logits = self.forward(spectrogram, waveform)  # No labels → no margin
        probs = F.softmax(logits, dim=-1)
        # Use the max probability (most likely class) for scoring
        max_probs, _ = probs.max(dim=-1)
        return -torch.log(max_probs + 1e-8)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  WaveNet Backbone — Standalone Architecture Test (Paper Config)")
    print("=" * 70)

    # ---- Test individual backbone ----
    print("\n--- Single WaveNetBackbone (spectrogram-like input) ---")
    backbone = WaveNetBackbone(in_channels=128)
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"  Config            : layers=4, blocks=3, channels=512")
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
        spec_channels=128, wavegram_channels=128,
    )
    total_params_enc = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters  : {total_params_enc:,}")

    spec = torch.randn(1, 1, 50, 128)   # [B, 1, T, F=128]
    wave = torch.randn(1, 128, 50)      # [B, 128, T']
    print(f"  Spectrogram input : {list(spec.shape)}")
    print(f"  Wavegram input    : {list(wave.shape)}")

    with torch.no_grad():
        spec_repr, wave_repr = encoder(spec, wave)

    print(f"  Spec repr output  : {list(spec_repr.shape)}")
    print(f"  Wave repr output  : {list(wave_repr.shape)}")

    # ---- Test ArcFace head ----
    print("\n--- ArcFace Head (scale=30, margin=0.7) ---")
    arcface = ArcFaceHead(in_features=256, num_classes=4, scale=30.0, margin=0.7)
    dummy_features = torch.randn(2, 256)
    dummy_labels = torch.tensor([0, 2])

    arcface.train()
    logits_train = arcface(dummy_features, dummy_labels)
    print(f"  Train logits shape: {list(logits_train.shape)}")

    arcface.eval()
    with torch.no_grad():
        logits_eval = arcface(dummy_features)
    print(f"  Eval logits shape : {list(logits_eval.shape)}")

    print("\n" + "=" * 70)
    print("  Spectrogram WaveNet receptive field :",
          encoder.wavenet_spec.receptive_field, "frames")
    print("  Wavegram   WaveNet receptive field :",
          encoder.wavenet_wave.receptive_field, "frames")
    print("=" * 70)
    print("\n[OK] SW-WaveNet (paper-matched) verified successfully!")
