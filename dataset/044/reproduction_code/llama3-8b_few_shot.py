# Import necessary libraries
import os
import wget

# Download Japanese TTS model
model_path = "jvn.tar.gz"
wget.download(f"https://dl.fbaipublicfiles.com/mms/tts/{model_path}", out=os.getcwd())

# Run infer.py with the downloaded model
infer_py_script = "infer.py"
os.system(f"python {infer_py_script} -m {model_path}")

# Output should be empty text after filtering OOV
print("Expected output:")