import os
import subprocess
import tempfile

# Create temporary audio files for testing
def create_dummy_audio_files():
    file_paths = []
    for i in range(1, 11):
        temp_file = f"audio{i}.wav"
        with open(temp_file, "wb") as f:
            # Create a minimal WAV file (just enough for the script to recognize it)
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        file_paths.append(temp_file)
    return file_paths

# Run MMS ASR inference
def run_mms_inference(audio_files):
    cmd = [
        "python", "-u", "examples/mms/asr/infer/mms_infer.py",
        "--model", "../fairseq_resource/mms1b_all.pt",
        "--lang", "eng",
        "--audio"
    ] + audio_files
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    return stdout, stderr

def main():
    audio_files = create_dummy_audio_files()
    stdout, stderr = run_mms_inference(audio_files)
    
    # Print the output to verify order issue
    print("Command output:")
    print(stdout)
    
    # Check if output order matches input order
    file_names = [os.path.basename(f) for f in audio_files]
    for i, name in enumerate(file_names):
        if name not in stdout:
            print(f"File {name} not found in output")
        else:
            position = stdout.find(name)
            print(f"File {name} position in output: {position}")

if __name__ == "__main__":
    main()