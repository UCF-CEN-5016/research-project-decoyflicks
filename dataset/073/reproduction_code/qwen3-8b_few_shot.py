import torch
import deepspeed

# Simulate a scenario where the communication backend 'cdb' is not initialized
def train():
    # Assume this is part of a distributed training setup
    tensor = torch.randn(10)
    # 'cdb' is expected to be initialized by DeepSpeed, but it's None here
    cdb = None  # Simulating uninitialized state
    cdb.all_reduce(tensor)  # This will raise the AttributeError

if __name__ == "__main__":
    train()