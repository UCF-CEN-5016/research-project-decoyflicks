import torch

if torch.backends.mps.is_available():
    print("MPS available")
else:
    print("MPS not available")