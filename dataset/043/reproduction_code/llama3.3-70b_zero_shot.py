import os
import sys
import subprocess

def main():
    # Clone fairseq
    subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq.git'])

    # Install fairseq
    subprocess.run(['pip', 'install', '--editable', './fairseq'])

    # Change directory to examples/mms/tts
    os.chdir('./fairseq/examples/mms/tts')

    # Call infer.py
    subprocess.run(['python', 'infer.py', '--model-dir', 'model/', '--wav', 'test.wav', '--txt', 'Heute ist ein schöner Tag.'])

if __name__ == "__main__":
    main()