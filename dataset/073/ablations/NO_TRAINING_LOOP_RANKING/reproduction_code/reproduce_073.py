import os
import torch
import deepspeed
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core import cdb  # Importing cdb to avoid AttributeError

# Set environment variables for distributed training
os.environ['WORLD_SIZE'] = '4'
os.environ['RANK'] = '0'

# Define model parallel sizes
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 2

# Initialize the distributed process group
torch.distributed.init_process_group(backend='nccl')

# Initialize model parallel
initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

# Create a dummy tensor
dummy_tensor = torch.randn(2, 2, 1024).cuda()

# Attempt to call cdb.all_reduce
try:
    print(cdb)  # Check the state of cdb
    result = cdb.all_reduce(dummy_tensor, op=torch.distributed.ReduceOp.SUM)
except AttributeError as e:
    print(e)  # This will reproduce the bug if cdb is not initialized