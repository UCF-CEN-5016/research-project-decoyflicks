import torch
import torchaudio

def simulate_forced_align_issue():
    # Simulate the issue with minimal setup
    audio = torch.randn(16000)  # 1 second of fake audio at 16kHz
    text = "hello world"
    emission = torch.randn(1, 100, 5)  # Correct 3D emission (batch_size, time, classes)

    try:
        alignment = torchaudio.functional.forced_align(
            emission,
            torch.tensor([len(text.split())]),  # target lengths
            text.split(),  # tokens
            0  # blank index
        )
        print("Alignment:", alignment)
    except RuntimeError as e:
        print("Error:", e)
        print("Expected shape: (batch_size, input_length, num_classes)")
        print("Actual shape:", emission.shape)

if __name__ == "__main__":
    simulate_forced_align_issue()