import threading
import time
import random

# Shared container for inference outputs
inference_results = []

def _process_audio_task(file_id: int) -> None:
    """Simulate ASR inference for a single audio file and store the result."""
    # Simulate inference delay
    time.sleep(random.uniform(0.1, 0.5))

    # Simulate output generation
    output = f"Output for audio_{file_id}.wav"
    inference_results.append(output)

def run_inference_threads(num_files: int = 10) -> None:
    """Start threads to simulate inference over multiple audio files."""
    file_ids = list(range(num_files))

    threads: list[threading.Thread] = []
    for fid in file_ids:
        thread = threading.Thread(target=_process_audio_task, args=(fid,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def display_results(results: list) -> None:
    """Print inference results (order is not guaranteed)."""
    print("Inference results:")
    for res in results:
        print(res)

if __name__ == "__main__":
    run_inference_threads()
    display_results(inference_results)