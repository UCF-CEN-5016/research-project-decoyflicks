import torch
from mms_infer import MMSInfer

# Load model and setup
model = MMSInfer('mms1b_all.pt', 'eng')
model.load()

# Define audio files to infer
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav', ... , 'audio10.wav']

# Infer with multiple audio inputs
for file in audio_files:
    # Simulate reading audio file (e.g., using librosa)
    audio, _ = read_audio_file(file)

    # Run inference on the audio file
    output = model.infer(audio)

    print(f"Input: {file}")
    print(f"Output: {output}")

print("MMS Infer log:")
for i, file in enumerate(audio_files):
    print(f"/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/{1089-134686-000{i}.wav}")