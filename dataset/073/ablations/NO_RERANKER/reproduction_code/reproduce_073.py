import os
import torch
import deepspeed
from megatron.core.parallel_state import initialize_model_parallel, get_data_parallel_group

# Set environment variables for distributed training
os.environ['WORLD_SIZE'] = '4'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

# Initialize the distributed process group
torch.distributed.init_process_group(backend='nccl', world_size=4, rank=0)

# Define parallel sizes
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 2

# Initialize model parallelism
initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size, 
                          pipeline_model_parallel_size=pipeline_model_parallel_size)

# Create a dummy tensor
dummy_tensor = torch.randn(2, 2, 1024).cuda()

# Attempt to call a function that uses the global cdb object
try:
    cdb = None  # Simulating uninitialized global object
    cdb.all_reduce(dummy_tensor, op='sum')
except AttributeError as e:
    print(e)  # Expected output: 'NoneType' object has no attribute 'all_reduce'