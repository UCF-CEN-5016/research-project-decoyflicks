import torch
import deepspeed
from deepspeed.runtime.comm.comm import cdb

# Simulate the distributed environment failure
def reproduce_bug():
    # Initialize tensors like in the original code
    tensor = torch.randn(10).cuda()
    
    # Attempt all_reduce without proper initialization
    try:
        # This will fail because cdb is None
        output = deepspeed.comm.all_reduce(tensor)
        print("Success (unexpected!)")
    except Exception as e:
        print(f"Error reproduced: {e}")
        print("This matches the original error where cdb is None")

if __name__ == "__main__":
    reproduce_bug()