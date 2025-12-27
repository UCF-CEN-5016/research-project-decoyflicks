import torch
import torchaudio
import os
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config  # Importing the missing Wav2Vec2Config

# Create dummy audio file
sample_rate = 16000
duration = 5  # seconds
num_samples = sample_rate * duration
dummy_audio = torch.randn(1, num_samples)
torchaudio.save('audio.wav', dummy_audio, sample_rate)

# Create corresponding text file
with open('text.txt', 'w') as f:
    f.write('This is a test transcription.')

# Ensure output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Set parameters
audio_filepath = 'audio.wav'
text_filepath = 'text.txt'
lang = 'udm'
uroman_path = 'uroman/bin'

# Load model
cfg = Wav2Vec2Config()  # Assuming Wav2Vec2Config is properly defined
model = Wav2Vec2Model.build_model(cfg)

# Run forced alignment
try:
    # Simulate forced alignment process
    log_probs = model(dummy_audio)  # This should trigger the shape mismatch error
except RuntimeError as e:
    if 'log_probs must be 3-D (batch_size, input length, num classes)' in str(e):
        print("Caught expected RuntimeError:", e)
    else:
        print("Caught unexpected RuntimeError:", e)