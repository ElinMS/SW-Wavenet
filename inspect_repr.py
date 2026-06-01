import torch
from wavenet_model import SWWaveNetClassifier
from wavegram_net import WavegramNet

# Load model checkpoint
ckpt_path = r'checkpoints/best_model.pt'
model = SWWaveNetClassifier(WavegramNet(), num_classes=4)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load spectrogram and wavegram features
spec_path = r'output/spectrograms/normal_id_00_00000000.pt'
wave_path = r'output/wavegrams/normal_id_00_00000000.pt'

spec = torch.load(spec_path, map_location='cpu')
wave = torch.load(wave_path, map_location='cpu')

print('--- Input Feature Tensors ---')
print('Spectrogram shape :', list(spec.shape))
print('Wavegram shape    :', list(wave.shape))

# Run the model encoder branch to get vectors
with torch.no_grad():
    spec_repr, wave_repr = model.encoder(spec, wave)

print('\n--- Generated Representation Vectors ---')
print('Spectrogram Representation shape:', list(spec_repr.shape))
print('Spectrogram Representation values (first 15 elements):')
print(spec_repr[0, :15].tolist())

print('\nWavegram Representation shape   :', list(wave_repr.shape))
print('Wavegram Representation values (first 15 elements):')
print(wave_repr[0, :15].tolist())

# Concatenate both representation vectors to get the final representation
combined_repr = torch.cat([spec_repr, wave_repr], dim=-1)
print('\nCombined Representation Vector shape:', list(combined_repr.shape))
