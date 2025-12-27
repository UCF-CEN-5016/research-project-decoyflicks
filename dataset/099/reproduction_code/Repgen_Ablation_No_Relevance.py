import torch

# Define batch size and sequence length
batch_size = 32
sequence_length = 100

# Generate random input data
input_data = torch.randn(batch_size, sequence_length, 512)

# Import RotaryEmbedding from the appropriate module
from rotary_embedding_module import RotaryEmbedding  # Assuming this is where RotaryEmbedding is defined

# Initialize the RotaryEmbedding layer
rotary_embedding = RotaryEmbedding(512)

# Apply the rotary embedding to the input data
output = rotary_embedding(input_data)

# Verify that output tensor contains NaN values in the first position along the sequence dimension
assert torch.isnan(output[:, 0, :]).any()

# Import pynvml for GPU memory monitoring
import pynvml

# Initialize pynvml
pynvml.nvmlInit()

# Get handle to the GPU device
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get memory information for the GPU device
info = pynvml.nvmlDeviceGetMemoryInfo(handle)

# Define expected memory usage threshold (double the current usage)
expected_memory_usage = info.used * 2

# Assert that the actual memory usage exceeds the expected threshold
assert info.used > expected_memory_usage