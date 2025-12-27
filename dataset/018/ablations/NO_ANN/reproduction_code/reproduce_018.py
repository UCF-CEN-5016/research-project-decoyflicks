import os
import subprocess

# Ensure Python 3.8 or higher is installed
subprocess.run(["python", "--version"], check=True)

# Install pip if not already installed
subprocess.run(["sudo", "pacman", "-S", "python-pip"], check=True)

# Clone the TensorFlow models repository
subprocess.run(["git", "clone", "https://github.com/tensorflow/models.git"], check=True)

# Navigate to the object detection directory
os.chdir("models/research/object_detection")

# Create a virtual environment
subprocess.run(["python", "-m", "venv", "tfod2_env"], check=True)

# Activate the virtual environment
subprocess.run(["source", "tfod2_env/bin/activate"], shell=True)

# Upgrade pip to the latest version
subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)

# Attempt to install the TensorFlow Object Detection API
try:
    subprocess.run(["python", "-m", "pip", "install", "--use-feature=2020-resolver", "."], check=True)
except subprocess.CalledProcessError as e:
    print(e.output)

# Verify the installed pip version
subprocess.run(["pip", "--version"], check=True)

# Check the current Python version
subprocess.run(["python", "--version"], check=True)