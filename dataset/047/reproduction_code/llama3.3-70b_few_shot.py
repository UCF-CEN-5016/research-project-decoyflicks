import threading
import time
from fairseq import checkpoint_utils, tasks
from fairseq.logging import progress_bar
from fairseq.models import FairseqSpeechEncoderDecoder

# Define a minimal function to simulate MMS ASR inference
def infer_audio(model, audio_file):
    # Simulate inference time
    time.sleep(0.1)
    # Generate a random output (for demonstration purposes)
    output = f"Output for {audio_file}"
    return output

# Define a list of audio files
audio_files = [
    "1089-134686-0000.wav",
    "1089-134686-0001.wav",
    "1089-134686-0002.wav",
    "1089-134686-0003.wav",
    "1089-134686-0004.wav",
    "1089-134686-0005.wav",
    "1089-134686-0006.wav",
    "1089-134686-0007.wav",
    "1089-134686-0008.wav",
    "1089-134686-0009.wav",
]

# Define a list to store the outputs
outputs = [None] * len(audio_files)

# Define a function to perform inference and store the output
def worker(index):
    output = infer_audio(None, audio_files[index])
    outputs[index] = output

# Create and start threads for each audio file
threads = []
for i in range(len(audio_files)):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

# Print the outputs (which may be disordered)
for i, output in enumerate(outputs):
    print(f"Input: {audio_files[i]}")
    print(f"Output: {output}")
    print("===============")