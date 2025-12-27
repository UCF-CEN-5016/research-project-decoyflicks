import os
import subprocess

# Step 1: Open a terminal on an Arch Linux system.
# Step 2: Ensure Python 3 and pip are installed
subprocess.run(["python", "--version"], check=True)
subprocess.run(["pip", "--version"], check=True)

# Step 3: Check the version of pip
pip_version = subprocess.run(["pip", "--version"], capture_output=True, text=True)
print(pip_version.stdout)

# Step 4: Upgrade pip if necessary
subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"], check=True)

# Step 5: Clone the TensorFlow models repository
subprocess.run(["git", "clone", "https://github.com/tensorflow/models.git"], check=True)

# Step 6: Navigate to the object detection directory
os.chdir("models/research/object_detection")

# Step 7: Create a virtual environment
subprocess.run(["python", "-m", "venv", "tfod2_env"], check=True)

# Step 8: Activate the virtual environment
activate_script = os.path.join("tfod2_env", "bin", "activate")
subprocess.run(["source", activate_script], shell=True)

# Step 9: Install TensorFlow
subprocess.run(["pip", "install", "tensorflow"], check=True)

# Step 10: Install the required dependencies
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# Step 11: Attempt to install the local package
try:
    subprocess.run(["python", "-m", "pip", "install", "--use-feature=2020-resolver", "."], check=True)
except subprocess.CalledProcessError as e:
    print(e.stderr)

# Step 12: Check the current directory
current_directory = os.getcwd()
print(f"Current directory: {current_directory}")

# Step 13: Check the version of pip again
pip_version = subprocess.run(["pip", "--version"], capture_output=True, text=True)
print(pip_version.stdout)

# Step 14: Document the steps taken and the error encountered