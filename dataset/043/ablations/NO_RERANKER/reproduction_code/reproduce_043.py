import os
import subprocess
import torch

# Step 1: Clone the fairseq repository
subprocess.run(["git", "clone", "https://github.com/pytorch/fairseq.git"])

# Step 2: Change directory to the cloned fairseq repository
os.chdir("fairseq")

# Step 3: Install fairseq in editable mode
subprocess.run(["pip", "install", "--editable", "./"])

# Step 4: Change directory to examples/mms/tts/
os.chdir("examples/mms/tts/")

# Step 5: Create a test.wav file
sample_rate = 22050
duration = 1  # seconds
t = torch.arange(0, sample_rate * duration)
wave = 0.5 * torch.sin(2 * 3.14159 * 440 * t / sample_rate)
torch.save(wave, "test.wav")

# Step 6: Create a text file named test.txt
with open("test.txt", "w") as f:
    f.write("Heute ist ein schöner Tag.")

# Step 7: Run the infer.py script
subprocess.run(["python", "infer.py", "--model-dir", "model/", "--wav", "test.wav", "--txt", "Heute ist ein schöner Tag."])

# Step 8: Check for ModuleNotFoundError
try:
    subprocess.run(["python", "infer.py", "--model-dir", "model/", "--wav", "test.wav", "--txt", "Heute ist ein schöner Tag."], check=True)
except subprocess.CalledProcessError as e:
    print(e)