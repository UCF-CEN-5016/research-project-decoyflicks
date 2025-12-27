import torch
import fairseq
import os
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(os.getcwd(), '../fairseq_resource/mms1b_all.pt')
lang = 'eng'
audio_files = glob.glob('/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/*.wav')
audio_files = audio_files[:10]
command = f'python -u examples/mms/asr/infer/mms_infer.py --model {model_path} --lang {lang} --audio ' + ' '.join(audio_files)
os.system(command)

log_file_path = 'path_to_output_log_file'  # Specify the path to the output log file
expected_file_path = 'path_to_expected_output_file'  # Specify the path to the expected output file

def extract_output_text(log_file_path):
    with open(log_file_path, 'r') as f:
        return f.read()

def read_expected_output_text(expected_file_path):
    with open(expected_file_path, 'r') as f:
        return f.read()

output_text = extract_output_text(log_file_path)
expected_text = read_expected_output_text(expected_file_path)
output_lines = output_text.split('\n')
expected_lines = expected_text.split('\n')
audio_to_expected = {f'1089-134686-{str(i).zfill(4)}.wav': expected_lines[i] for i in range(len(expected_lines))}

for i, audio_file in enumerate(audio_files):
    assert output_lines[i].startswith(audio_to_expected[audio_file.split('/')[-1]])
    if output_lines[i] != audio_to_expected[audio_file.split('/')[-1]]:
        print(f'Output mismatch for {audio_file}: {output_lines[i]} != {audio_to_expected[audio_file.split("/")[-1]]}')