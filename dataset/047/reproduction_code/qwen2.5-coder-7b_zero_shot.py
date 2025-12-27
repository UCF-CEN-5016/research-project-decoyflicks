import os
from fairseq.mms import MMS

BASE_INPUT_DIR = '/Dataset/speech/english/test/libri_test_other/wav/wav/1/wav/1/'

def format_input_path(base_dir, filename):
    return os.path.join(base_dir, filename)

def process_audio_files(model, filenames, base_dir):
    for name in filenames:
        input_path = format_input_path(base_dir, name)
        print(f"Input: {input_path}")
        output = model.infer(name)
        print(f"Output: {output}")

def main():
    model = MMS(model_path='path/to/model.pt', lang='eng')
    audio_list = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    process_audio_files(model, audio_list, BASE_INPUT_DIR)

if __name__ == '__main__':
    main()