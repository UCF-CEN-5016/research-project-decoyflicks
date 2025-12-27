import torch
import fairseq
import os
import glob

model_path = os.path.join(os.getcwd(), '../fairseq_resource/mms1b_all.pt')
lang = 'eng'
audio_files = glob.glob('/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/*.wav')
assert len(audio_files) >= 10, 'Not enough audio files for testing'
selected_audio_files = audio_files[:10]
command = f'python -u examples/mms/asr/infer/mms_infer.py --model {model_path} --lang {lang} --audio {" ".join(selected_audio_files)}'
os.system(command)

output_log = open('output.log', 'r').readlines()
expected_output = ['1089-134686-0000', '1089-134686-0001', '1089-134686-0002', '1089-134686-0003', '1089-134686-0004', '1089-134686-0005', '1089-134686-0006', '1089-134686-0007', '1089-134686-0008', '1089-134686-0009']
actual_output = [line.split(' ')[-1].strip() for line in output_log if 'Input:' in line]
assert actual_output != expected_output, 'Output order is correct, but it should be incorrect to reproduce the bug'
print('Actual output order:', actual_output)
print('Expected output order:', expected_output)
assert not any('error' in line.lower() for line in output_log), 'Errors found in output log'