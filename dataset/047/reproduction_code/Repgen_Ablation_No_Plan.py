import os

def reproduce_bug():
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]  # Example audio files
    command = f"python -u examples/mms/asr/infer/mms_infer.py --model `pwd`/../fairseq_resource/mms1b_all.pt --lang \"eng\" {' '.join(audio_files)}"
    os.system(command)

if __name__ == "__main__":
    reproduce_bug()