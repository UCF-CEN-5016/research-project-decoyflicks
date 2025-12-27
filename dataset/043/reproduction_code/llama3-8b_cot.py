import os
import pip

# Set up minimal environment
os.chdir('examples/mms/tts/')
pip.main(['install', '--editable', './'])

# Add triggering conditions
from fairseq import infer  # This line should trigger the bug
infer.py --model-dir model/ --wav test.wav --txt "Heute ist ein schöner Tag."