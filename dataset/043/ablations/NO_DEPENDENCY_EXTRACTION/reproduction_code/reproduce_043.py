import os
import subprocess

# Step 1: Clone the fairseq repository
subprocess.run(["git", "clone", "https://github.com/pytorch/fairseq.git"], check=True)

# Step 2: Change directory to the cloned fairseq repository
os.chdir("fairseq")

# Step 3: Install fairseq in editable mode
subprocess.run(["pip", "install", "--editable", "./"], check=True)

# Step 4: Change directory to examples/mms/tts
os.chdir("examples/mms/tts")

# Step 5: Create a test.wav file
subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", "test.wav"], check=True)

# Step 6: Prepare the text input
text_input = "Heute ist ein schöner Tag."

# Step 7: Run the infer.py script
subprocess.run(["python", "infer.py", "--model-dir", "model/", "--wav", "test.wav", "--txt", text_input], check=True)

# Step 8: Check for ModuleNotFoundError
try:
    subprocess.run(["python", "infer.py", "--model-dir", "model/", "--wav", "test.wav", "--txt", text_input], check=True)
except subprocess.CalledProcessError as e:
    print(e)