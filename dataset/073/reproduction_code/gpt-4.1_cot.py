# reproduction.py

# Step 1: Necessary imports
import torch

# Step 2: Minimal environment setup
# Simulate the global cdb which should be initialized but is None
cdb = None

def run_all_reduce(tensor):
    # Step 3: Triggering condition - calling all_reduce on None
    return cdb.all_reduce(tensor)

if __name__ == "__main__":
    tensor = torch.tensor([1, 2, 3])

    # This will raise the AttributeError as cdb is None
    run_all_reduce(tensor)