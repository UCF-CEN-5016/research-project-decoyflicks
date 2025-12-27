# This script directly attempts to run the infer.py script that causes the error
import subprocess
import sys
import os

def main():
    # Navigate to the examples/mms/tts directory
    os.chdir("examples/mms/tts")
    
    # Execute the command that produces the error
    try:
        subprocess.run(
            ["python", "infer.py", "--model-dir", "model/", "--wav", "test.wav", "--txt", "Heute ist ein schöner Tag."],
            check=True
        )
        print("No error occurred, which is unexpected")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
    
if __name__ == "__main__":
    main()