import torch
import torchaudio

def reproduce_bug():
    waveform = torch.randn(1, 16000)  # Fake audio data
    transcript = "test text"
    emission = torch.randn(1, 100, 10)  # Incorrect shape log_probs
    
    try:
        segments = torchaudio.functional.forced_align(
            emission,
            torch.tensor([len(transcript.split())]),
            transcript.split(),
            torch.tensor([waveform.shape[1]]),
        )
    except RuntimeError as e:
        print(f"Error reproduced: {e}")

reproduce_bug()