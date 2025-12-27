import tarfile
import wget
import torch
import numpy as np

wget.download('https://dl.fbaipublicfiles.com/mms/tts/jvn.tar.gz')
tar = tarfile.open('jvn.tar.gz')
tar.extractall()
tar.close()

import os
os.chdir('jvn')

from glow_tts import GlowTTS
model = GlowTTS('config.json', 'glow_tts.pt')

text = 'Hello, how are you?'
oov_dict = model.filter_oov(text)

print('text after filtering OOV:', oov_dict['text'])