import torch
import deepspeed

# Normally you should call deepspeed.init_distributed() here,
# but we skip it to simulate the bug.

# Create a tensor on CUDA device 0
tensor = torch.ones(1).cuda()

# Try to all_reduce without initializing deepspeed distributed communication
try:
    deepspeed.comm.all_reduce(tensor)
except Exception as e:
    print(f"Caught exception: {e}")