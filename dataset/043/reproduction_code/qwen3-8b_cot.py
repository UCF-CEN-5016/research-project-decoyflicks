import os
import sys

# Step 1: Set up environment
# Clone fairseq repository
os.system("git clone https://github.com/facebookresearch/fairseq.git")

# Step 2: Install in editable mode
os.chdir("fairseq")
os.system("pip install --editable .")

# Step 3: Navigate to the TTS example directory
os.chdir("examples/mms/tts")

# Step 4: Run the infer.py script
# The script should trigger the ModuleNotFoundError
os.system("python infer.py --model-dir model/ --wav test.wav --txt \"Heute ist ein schöner Tag.\"")

from fairseq import commons

import sys
sys.path.append("..")  # Or the correct relative path to the fairseq root
import commons