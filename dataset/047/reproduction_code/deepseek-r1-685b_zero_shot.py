import time
import random
from concurrent.futures import ThreadPoolExecutor

def process_audio(audio_path):
    # Simulate processing time variability
    time.sleep(random.uniform(0.1, 0.5))
    return f"Output for {audio_path}"

def main(audio_files):
    print(">>> loading model & running inference ...")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_audio, audio_files))
    
    for path, result in zip(audio_files, results):
        print("===============")
        print(f"Input: {path}")
        print(f"Output: {result}")

if __name__ == "__main__":
    audio_files = [f"audio{i}.wav" for i in range(10)]
    print("\n".join(audio_files))
    main(audio_files)