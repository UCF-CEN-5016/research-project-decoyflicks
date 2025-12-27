import torch
import torchaudio
from torchaudio.models import wav2vec2_model  # or any ASR model
from torchaudio.functional import forced_align

def main():
    # Simulate a batch of audio waveforms (batch_size=1, waveform length=16000)
    waveform = torch.randn(1, 16000)

    # Simulate a transcript (text to token IDs)
    # For simplicity, assume tokens from 0 to N-1 where N = number of classes
    num_classes = 10
    tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)

    # Simulate log_probs with wrong shape (e.g., 2D instead of 3D)
    # Correct shape should be (batch_size, seq_len, num_classes)
    # Let's create a 2D tensor (seq_len, num_classes)
    seq_len = 50
    log_probs_wrong = torch.randn(seq_len, num_classes).log_softmax(dim=-1)

    try:
        # This should raise an error because log_probs is not 3D
        alignment = forced_align(
            log_probs=log_probs_wrong,
            transcript=tokens,
            blank=0,
        )
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")

    # Now fix the shape by adding batch dimension
    log_probs_correct = log_probs_wrong.unsqueeze(0)  # shape: (1, seq_len, num_classes)

    # This should work or produce different error related to content but not shape
    try:
        alignment = forced_align(
            log_probs=log_probs_correct,
            transcript=tokens,
            blank=0,
        )
        print("Forced alignment succeeded with correctly shaped log_probs.")
    except Exception as e:
        print(f"Got error with correct shape: {e}")

if __name__ == "__main__":
    main()