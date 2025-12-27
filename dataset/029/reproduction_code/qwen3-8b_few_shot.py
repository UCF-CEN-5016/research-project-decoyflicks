import torch

# Check for MPS availability (which may not be available in some PyTorch versions)
if torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")