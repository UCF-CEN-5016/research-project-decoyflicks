import torch
import deepspeed
import argparse
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description='Reproduce DeepSpeed all_reduce bug')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank passed from distributed launcher')

# Add DeepSpeed arguments
deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Print environment variables for debugging
print(f"RANK: {os.environ.get('RANK', 'Not set')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")

# Initialize DeepSpeed distributed environment
print("Initializing distributed environment...")
try:
    deepspeed.init_distributed()
    print("DeepSpeed distributed initialization successful")
except Exception as e:
    print(f"DeepSpeed distributed initialization failed: {e}")

# Create a simple model
print("Creating model...")
model = torch.nn.Linear(10, 10)

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 8,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    }
}

# Initialize DeepSpeed engine
print("Initializing DeepSpeed engine...")
try:
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    print("DeepSpeed engine initialization successful")
except Exception as e:
    print(f"DeepSpeed engine initialization failed: {e}")

# Create a dummy input and perform forward pass
print("Running forward pass...")
dummy_input = torch.randn(8, 10).to(model_engine.device)
outputs = model_engine(dummy_input)

# Perform all_reduce operation directly to trigger the bug
print("Attempting all_reduce operation...")
try:
    # This is the operation that might fail with 'NoneType' object has no attribute 'all_reduce'
    # Access the communication backend directly
    comm = model_engine.comm
    print(f"Communication backend type: {type(comm)}")
    
    # Create a tensor to reduce
    tensor = torch.ones(1).to(model_engine.device)
    print(f"Before all_reduce: {tensor}")
    
    # Attempt the all_reduce operation
    comm.all_reduce(tensor)
    print(f"After all_reduce: {tensor}")
    
    print("All_reduce operation successful")
except Exception as e:
    print(f"All_reduce operation failed with error: {e}")

print("Script completed")