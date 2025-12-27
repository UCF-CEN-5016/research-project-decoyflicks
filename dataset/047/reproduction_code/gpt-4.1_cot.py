import concurrent.futures
import time
import random

# Dummy inference function simulating variable processing time
def infer(audio_path):
    # Simulate variable inference time
    time_to_sleep = random.uniform(0.1, 0.5)
    time.sleep(time_to_sleep)
    # Return dummy transcription (last part of filename reversed for demo)
    transcript = f"Transcript for {audio_path.split('/')[-1]}"
    return audio_path, transcript

def main():
    # List of dummy audio file paths in order
    audio_files = [f"audio_{i:02d}.wav" for i in range(10)]

    print("=== Running inference and printing output immediately (unordered) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(infer, audio): audio for audio in audio_files}
        for future in concurrent.futures.as_completed(futures):
            audio_path, transcript = future.result()
            print(f"{audio_path}: {transcript}")

    print("\n=== Running inference and printing output ordered by input list ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(infer, audio) for audio in audio_files]
        results = [future.result() for future in futures]

    # Now results are collected in input order, print in order
    for audio_path, transcript in results:
        print(f"{audio_path}: {transcript}")

if __name__ == "__main__":
    main()