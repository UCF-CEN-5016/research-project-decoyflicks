import torch
import torchaudio
import os
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

# Create dummy audio file
sample_rate = 16000
duration = 5  # seconds
num_samples = sample_rate * duration
dummy_audio = torch.randn(1, num_samples)
torchaudio.save('audio.wav', dummy_audio, sample_rate)

# Create corresponding text file
with open('text.txt', 'w') as f:
    f.write('This is a test transcription.')

# Set parameters
audio_filepath = 'audio.wav'
text_filepath = 'text.txt'
lang = 'udm'
outdir = 'output'
uroman_path = 'uroman/bin'

# Ensure output directory exists
os.makedirs(outdir, exist_ok=True)

# Run forced alignment command
os.system(f'python align_and_segment.py --audio_filepath {audio_filepath} --text_filepath {text_filepath} --lang {lang} --outdir {outdir} --uroman {uroman_path}')

# Check for log_probs shape mismatch
log_probs = torch.randn(1, 10, 5)  # Example shape that is not 3-D
print(f'log_probs shape: {log_probs.shape}')
assert log_probs.dim() == 3, "log_probs must be 3-D (batch_size, input length, num classes)"