import torch
import torchaudio

def simulate_alignment_issue():
    # Simulate the issue with minimal setup
    audio = torch.randn(16000)  # 1 second of fake audio at 16kHz
    text = "hello world"
    emission = torch.randn(100, 5)  # Incorrect 2D emission (time, classes)

    # Correct the shape of emission to be 3D
    emission = emission.unsqueeze(0)  # Convert to 3D (batch_size=1)

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

# Call the function to simulate the alignment issue
simulate_alignment_issue()