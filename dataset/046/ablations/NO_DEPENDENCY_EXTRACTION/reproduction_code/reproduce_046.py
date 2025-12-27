import torch
import torchaudio
import os
import subprocess

# Create dummy audio file
sample_rate = 16000
duration = 5  # seconds
num_samples = sample_rate * duration
waveform = torch.zeros((1, num_samples))  # Silent audio
torchaudio.save('audio.wav', waveform, sample_rate)

# Create corresponding text file
with open('text.txt', 'w') as f:
    f.write('This is a test transcription.')

# Set parameters
audio_filepath = 'audio.wav'
text_filepath = 'text.txt'
lang = 'udm'
outdir = 'output'
uroman_path = 'uroman/bin'

# Run the command
command = f'python align_and_segment.py --audio_filepath {audio_filepath} --text_filepath {text_filepath} --lang {lang} --outdir {outdir} --uroman {uroman_path}'
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Capture output and check for error
if 'RuntimeError: log_probs must be 3-D (batch_size, input length, num classes)' in result.stderr:
    print("Error reproduced:", result.stderr)
else:
    print("No error found.")