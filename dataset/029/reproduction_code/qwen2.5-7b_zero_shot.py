import torch

# Check if the Memory Persistence Service (MPS) is available
mps_available = torch.backends.mps.is_available()
print(mps_available)