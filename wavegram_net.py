import torch
import torch.nn as nn

class WavegramNet(nn.Module):
    """
    Extracts a learned 1D Wavegram from a raw 1D audio waveform.
    Matches the SW-WaveNet paper specification:
    A single 1D Convolutional layer with K=1024, Cout=128, and D=1.
    Stride is set to 512 to match log-mel spectrogram hop_length.
    """
    def __init__(self):
        super(WavegramNet, self).__init__()
        # Table 1: Cin=1, Cout=128, K=1024, D=1
        self.conv = nn.Conv1d(1, 128, kernel_size=1024, stride=512)

    def forward(self, x):
        """
        Args:
            x: Raw waveform tensor of shape [Batch, 1, Time]
        Returns:
            wavegram: Learned representation of shape [Batch, 128, Time']
        """
        return self.conv(x)

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
    print(f"  -> [Batch={wavegram.shape[0]}, C={wavegram.shape[1]}, T={wavegram.shape[2]}]")

    print("\nWavegramNet architecture verified successfully!")
