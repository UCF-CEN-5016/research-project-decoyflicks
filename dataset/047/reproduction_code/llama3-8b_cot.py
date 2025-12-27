import torch
from transformers import MMSForASR

import os
from mms_infer import main

# Set the model path and language
model_path = 'path/to/mms1b_all.pt'
lang = 'eng'

# Define the audio files to use for inference
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav', ... , 'audio10.wav']

if __name__ == '__main__':
    main(model_path, lang, audio_files)

import os
from transformers import MMSForASR

# Set the model path and language
model_path = 'path/to/mms1b_all.pt'
lang = 'eng'

# Define the audio files to use for inference
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav', ... , 'audio10.wav']

if __name__ == '__main__':
    main(model_path, lang, audio_files)