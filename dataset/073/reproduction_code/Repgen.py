import torch
import deepspeed
import argparse

# Parse arguments (required for DeepSpeed)
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

# Initialize DeepSpeed but deliberately create a scenario where communication backend isn't properly set up
deepspeed.init_distributed()

# Create a model
model = torch.nn.Linear(10, 10)

# Initialize DeepSpeed with a configuration that might lead to issues
ds_config = {
    "train_batch_size": 8,
    "fp16": {"enabled": True},
}

# Create a dummy tensor we'll try to all_reduce
tensor = torch.ones(1)

# Simulate the global cdb object not being initialized properly
# In a real scenario, this happens due to configuration issues or improper initialization
cdb = None

try:
    # This will fail with 'NoneType' object has no attribute 'all_reduce'
    print("Attempting all_reduce operation...")
    cdb.all_reduce(tensor)
except AttributeError as e:
    print(f"Error: {e}")
    print("This reproduces the bug where 'cdb' is None but 'all_reduce' is called on it")