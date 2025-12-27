import os
from fairseq.mms import MMS

mms = MMS(model_path='path/to/model.pt', lang='eng')
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']

for file in audio_files:
    print(f"Input: {os.path.join('/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/', file)}")
    output = mms.infer(file)
    print(f"Output: {output}")