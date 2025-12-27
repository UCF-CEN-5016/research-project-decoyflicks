import torch
import torchaudio

def simulate_alignment_issue():
    # Simulate the issue with minimal setup
    audio = torch.randn(16000)  # 1 second of fake audio at 16kHz
    text = "hello world"
    emission = torch.randn(1, 100, 5)  # Correct 3D emission (batch_size, time, classes)

    # This will raise the RuntimeError
    try:
        alignment = torchaudio.functional.forced_align(
            emission,
            torch.tensor([len(text.split())]),
            text.split(),
            0
        )
        print("Alignment:", alignment)
    except RuntimeError as e:
        print("Error:", e)
        print("Expected shape: (batch_size, input_length, num_classes)")
        print("Actual shape:", emission.shape)

simulate_alignment_issue()