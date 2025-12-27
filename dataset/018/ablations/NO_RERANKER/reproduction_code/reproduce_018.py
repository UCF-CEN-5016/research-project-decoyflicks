import os
import subprocess
import sys

def main():
    # Step 1: Open a terminal on an Arch Linux system.
    
    # Step 2: Ensure Python 3.8 or higher is installed
    python_version = subprocess.check_output(["python", "--version"]).decode().strip()
    print(f"Python version: {python_version}")

    # Step 3: Install pip if not already installed
    subprocess.run(["sudo", "pacman", "-S", "python-pip"], check=True)

    # Step 4: Clone the TensorFlow models repository
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models.git"], check=True)

    # Step 5: Navigate to the object detection directory
    os.chdir("models/research/object_detection")

    # Step 6: Create a virtual environment
    subprocess.run(["python", "-m", "venv", "tfod2_env"], check=True)

    # Step 7: Activate the virtual environment
    activate_script = os.path.join("tfod2_env", "bin", "activate")
    subprocess.run(["source", activate_script], shell=True)

    # Step 8: Upgrade pip to the latest version
    subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)

    # Step 9: Attempt to install the TensorFlow Object Detection API
    try:
        subprocess.run(["python", "-m", "pip", "install", "--use-feature=2020-resolver", "."], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Step 10: Verify the installed pip version
    pip_version = subprocess.check_output(["pip", "--version"]).decode().strip()
    print(f"Pip version: {pip_version}")

    # Step 11: Check the current Python version
    python_version = subprocess.check_output(["python", "--version"]).decode().strip()
    print(f"Current Python version: {python_version}")

    # Step 12: Confirm the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

if __name__ == "__main__":
    main()