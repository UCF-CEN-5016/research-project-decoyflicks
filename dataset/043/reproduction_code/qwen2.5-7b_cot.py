import os

def clone_repository(repo_url):
    os.system(f"git clone {repo_url}")

def install_package():
    os.system("pip install --editable ./fairseq")

def navigate_to_directory(directory):
    os.chdir(directory)

def run_inference_script(script_path, model_dir, wav_file, text):
    os.system(f"python {script_path} --model-dir {model_dir} --wav {wav_file} --txt \"{text}\"")

if __name__ == "__main__":
    # Step 1: Set up environment
    clone_repository("https://github.com/facebookresearch/fairseq.git")

    # Step 2: Install in editable mode
    os.chdir("fairseq")
    install_package()

    # Step 3: Navigate to the TTS example directory
    navigate_to_directory("examples/mms/tts")

    # Step 4: Run the infer.py script
    run_inference_script("infer.py", "model/", "test.wav", "Heute ist ein schöner Tag.")

    from fairseq import commons

    sys.path.append("..")  # Or the correct relative path to the fairseq root
    import commons