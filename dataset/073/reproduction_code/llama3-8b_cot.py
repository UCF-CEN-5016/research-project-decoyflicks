import torch
from deepspeed import init_dataloader

# Initialize a dummy dataset using dataloader
dummy_dataset = [{"input_ids": [1, 2, 3], "labels": [0, 1, 2]} for _ in range(10)]
data_loader = init_dataloader(dummy_dataset, batch_size=2)

# Try to run the script with an incorrect cdb initialization
try:
    cdb = None  # Set cdb to None to reproduce the error
    data_loader.all_reduce()  # This should raise the AttributeError
except Exception as e:
    print(f"Error: {e}")