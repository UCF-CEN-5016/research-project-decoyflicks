import torch
import torchaudio
import os
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.data import Dictionary

# Setup
audio_filepath = 'audio.wav'
text_filepath = 'text.txt'
lang = 'udm'
outdir = 'output'
uroman_path = 'uroman/bin'

# Create dummy audio file
sample_rate = 16000
duration = 5  # seconds
num_samples = sample_rate * duration
dummy_audio = torch.randn(1, num_samples)
torchaudio.save(audio_filepath, dummy_audio, sample_rate)

# Create corresponding text file
with open(text_filepath, 'w') as f:
    f.write('This is a test transcription.')

# Run forced alignment command
os.system(f'python align_and_segment.py --audio_filepath {audio_filepath} --text_filepath {text_filepath} --lang {lang} --outdir {outdir} --uroman {uroman_path}')

# Check for error message
try:
    # Assuming the log_probs tensor is generated in the align_and_segment.py script
    # Add print statements in the relevant part of the code where log_probs is generated
    log_probs = torch.randn(2, 10)  # Simulating incorrect shape
    assert log_probs.shape != (2, 10, 5)  # Example shape mismatch
except RuntimeError as e:
    print(e)