import torch

# Attempt to check for MPS availability, which is not supported on Linux systems
if torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")