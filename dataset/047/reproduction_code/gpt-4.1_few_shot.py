import threading
import time
import random

# Simulated ASR inference function with variable latency
def asr_infer(audio_path, results, index):
    # Simulate variable processing time per audio file
    time.sleep(random.uniform(0.1, 0.5))
    # Fake transcription output (just echoing index for demo)
    transcription = f"Transcription for {audio_path}"
    results[index] = (audio_path, transcription)
    print(f"Input: {audio_path}")
    print(f"Output: {transcription}")

# List of audio files in order
audio_files = [f"audio_{i}.wav" for i in range(10)]

# Dictionary to store results by index
results = {}

# Launch inference threads for each audio file
threads = []
for idx, audio in enumerate(audio_files):
    t = threading.Thread(target=asr_infer, args=(audio, results, idx))
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

print("\nResults dictionary (ordered by index):")
for i in range(len(audio_files)):
    print(f"{results[i][0]} -> {results[i][1]}")

# Although results dict preserves order, the print statements inside threads happen in completion order,
# causing the output logs to appear out of order relative to input list.