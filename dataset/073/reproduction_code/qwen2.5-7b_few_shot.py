import torch
import deepspeed

def train():
    # Assume this is part of a distributed training setup
    tensor = torch.randn(10)
    # Initialize the communication backend 'cdb' using DeepSpeed
    cdb = deepspeed.CPUoffload()
    cdb.all_reduce(tensor)

if __name__ == "__main__":
    train()