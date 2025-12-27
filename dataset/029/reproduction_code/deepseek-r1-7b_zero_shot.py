import torch
from torch import distributed as dist

# Set device ID (adjust based on CUDA availability)
local_rank = 0
if torch.cuda.is_available():
    local_rank = int(local_rank)  # Use 0 if using CPU, or set to desired GPU ID if available

print(f"Using device: {torch.cuda.get_device_name(local_rank)}")