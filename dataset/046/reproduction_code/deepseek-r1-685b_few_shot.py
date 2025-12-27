import torch
import torchaudio

# Simulate the issue with minimal setup
audio = torch.randn(16000)  # 1 second of fake audio at 16kHz
text = "hello world"
emission = torch.randn(100, 5)  # Incorrect 2D emission (time, classes)

# This will raise the RuntimeError
try:
    alignment = torchaudio.functional.forced_align(
        emission,  # Should be 3D
        torch.tensor([len(text.split())]),  # target lengths
        text.split(),  # tokens
        0  # blank index
    )
    print("Alignment:", alignment)
except RuntimeError as e:
    print("Error:", e)
    print("Expected shape: (batch_size, input_length, num_classes)")
    print("Actual shape:", emission.shape)