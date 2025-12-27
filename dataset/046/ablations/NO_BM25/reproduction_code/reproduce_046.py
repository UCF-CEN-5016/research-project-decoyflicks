import torch
import torchaudio
import subprocess

def load_audio(audio_filepath):
    audio, sample_rate = torchaudio.load(audio_filepath)
    assert audio.shape[0] == 1, "Audio must have one channel"
    return audio

def read_transcription(text_filepath):
    with open(text_filepath, 'r') as f:
        text = f.read().strip()
    assert text, "Transcription must not be empty"
    return text

audio_filepath = 'audio.wav'
text_filepath = 'text.txt'
lang = 'udm'
outdir = 'output'
uroman_path = 'uroman/bin'

audio = load_audio(audio_filepath)
transcription = read_transcription(text_filepath)

command = f'python align_and_segment.py --audio_filepath {audio_filepath} --text_filepath {text_filepath} --lang {lang} --outdir {outdir} --uroman {uroman_path}'
result = subprocess.run(command, shell=True, capture_output=True, text=True)

if "RuntimeError: log_probs must be 3-D (batch_size, input length, num classes)" in result.stderr:
    print("Bug reproduced: log_probs shape error found.")
    print(f"Input tensor shape: {audio.shape}")