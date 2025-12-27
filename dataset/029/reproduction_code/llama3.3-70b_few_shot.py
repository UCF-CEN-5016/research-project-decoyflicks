import torch

# Check if 'mps' is available
if torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")

# This will cause the AttributeError
try:
    torch.backends.mps
except AttributeError as e:
    print(f"AttributeError: {e}")