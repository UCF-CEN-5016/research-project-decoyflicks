import torch
from fairseq.models.wav2vec import Wav2VecModel
from pathlib import Path
import soundfile as sf

# Mock setup similar to MMS inference
class MMSInfer:
    def __init__(self, model_path):
        self.model = Wav2VecModel.from_pretrained(model_path)
        self.model.eval()
        
    def infer(self, audio_paths):
        results = []
        for path in audio_paths:
            # Simulate async behavior that could cause ordering issues
            audio, _ = sf.read(path)
            with torch.no_grad():
                # Actual inference would happen here
                # Mock output generation
                output = f"transcript for {Path(path).name}"
                results.append((path, output))
        return results

# Test case that reproduces the issue
def test_disordered_output():
    # Create mock audio files
    audio_files = [f"audio{i}.wav" for i in range(10)]
    
    # Initialize with dummy model path
    infer = MMSInfer("dummy_model.pt")
    
    # Get results - in actual bug these come out of order
    results = infer.infer(audio_files)
    
    # Print results to show mismatch
    print("Input order vs Output order:")
    for i, (path, trans) in enumerate(results):
        print(f"Input {i}: {Path(path).name} -> Output: {trans}")

if __name__ == "__main__":
    test_disordered_output()

from concurrent.futures import ThreadPoolExecutor
import numpy as np

def infer_ordered(self, audio_paths):
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, path in enumerate(audio_paths):
            future = executor.submit(self._process_one, idx, path)
            futures.append(future)
        
        # Get results in completion order but sort by original index
        results = [f.result() for f in futures]
        results.sort(key=lambda x: x[0])  # sort by original index
        return [r[1] for r in results]  # return just transcripts in order