# This script attempts to reproduce the exact steps from the bug report
import subprocess
import os
import sys

def main():
    # Step 1: Clone the fairseq repository if it doesn't exist
    if not os.path.exists("fairseq"):
        print("Cloning fairseq repository...")
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/fairseq.git"], check=True)
        os.chdir("fairseq")
    elif os.path.exists("fairseq"):
        os.chdir("fairseq")
    
    # Step 2: Install the project in editable mode
    print("Installing fairseq in editable mode...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--editable", "."], check=True)
    
    # Step 3: Navigate to the examples/mms/tts directory
    if os.path.exists("examples/mms/tts"):
        os.chdir("examples/mms/tts")
        print("Changed directory to examples/mms/tts")
    else:
        print("Directory examples/mms/tts not found")
        return
    
    # Step 4: Execute the command from the bug report
    print("Running the infer.py script...")
    try:
        subprocess.run(
            [sys.executable, "infer.py", "--model-dir", "model/", "--wav", "test.wav", "--txt", "Heute ist ein schöner Tag."],
            check=True
        )
        print("Command executed successfully, which is unexpected")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}, which is expected")
    except FileNotFoundError:
        print("infer.py not found")
    
if __name__ == "__main__":
    main()