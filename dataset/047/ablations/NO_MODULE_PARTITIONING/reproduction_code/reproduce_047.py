import torch
import fairseq
import os
import glob
import subprocess

# Set device for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model path and language
model_path = os.path.join(os.getcwd(), '../fairseq_resource/mms1b_all.pt')
lang = 'eng'

# Gather audio files
audio_files = glob.glob('/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/*.wav')
assert len(audio_files) >= 10, 'Not enough audio files'
audio_files = audio_files[:10]

# Prepare command for ASR inference
command = f'python -u examples/mms/asr/infer/mms_infer.py --model {model_path} --lang {lang} --audio {" ".join(audio_files)}'

# Execute the command and capture the output
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
logs, errors = process.communicate()

# Write logs to a file for later analysis
with open('inference_log.txt', 'wb') as f:
    f.write(logs)

# Function to extract output text from logs (to be defined)
def extract_output_text(logs):
    # Placeholder for actual extraction logic
    return logs.decode('utf-8').splitlines()

# Function to read reference text from a file (to be defined)
def read_reference_text(reference_path):
    with open(reference_path, 'r') as f:
        return f.read().splitlines()

# Extract output text and read expected text
output_text = extract_output_text(logs)
expected_text = read_reference_text('/path/to/reference_text.txt')

# Assert that the output order matches the expected order
assert output_text == expected_text, 'Output log order is incorrect'

# Print output and expected text for comparison
print(output_text, expected_text)