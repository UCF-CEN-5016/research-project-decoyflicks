import torch

# Attempt to check for MPS (Apple Silicon GPU) availability on PyTorch 1.9.0
try:
    if torch.backends.mps.is_available():
        print("MPS backend is available")
    else:
        print("MPS backend is not available")
except AttributeError as e:
    print("Caught AttributeError:", e)