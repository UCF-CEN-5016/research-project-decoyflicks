import threading
import time
import random

# Simulate ASR inference with multiple audio files
results = []

def process_audio(file_id):
    # Simulate inference delay
    time.sleep(random.uniform(0.1, 0.5))
    
    # Simulate output generation
    output = f"Output for audio_{file_id}.wav"
    results.append(output)

def start_threads():
    # Simulate multiple audio files (audio_0.wav to audio_9.wav)
    audio_files = list(range(10))
    
    # Start inference threads
    threads = []
    for file_id in audio_files:
        t = threading.Thread(target=process_audio, args=(file_id,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

def print_results(results):
    # Print results (order is not guaranteed)
    print("Inference results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    start_threads()
    print_results(results)