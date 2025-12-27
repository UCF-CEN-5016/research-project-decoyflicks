import torch
import torch.nn as nn

# Import custom rotary_embedding_torch module
from rotary_embedding_torch import RotaryEmbedding

# Define dimensions for Q and K matrices
dim = 512
seq_len = 64

# Create a batch of random input data for Q with shape (batch_size=32, seq_len, dim)
Q = torch.randn(32, seq_len, dim)

# Create a batch of random input data for K with the same shape as Q
K = torch.randn(32, seq_len, dim)

# Initialize an instance of RotaryEmbedding with use_xpos=True and xpos_scale_base=512
rotary_embedding = RotaryEmbedding(dim, use_xpos=True, xpos_scale_base=512)

# Call rotate_queries_and_keys method on the initialized RotaryEmbedding instance with Q and K as inputs
rotated_q, rotated_k = rotary_embedding.rotate_queries_and_keys(Q, K)

# Verify that rotated_k contains a very small fraction of NaN values
assert torch.isnan(rotated_k).sum() > 0

# Calculate a loss function (e.g., mean squared error between Q and rotated_q)
loss = nn.MSELoss()(Q, rotated_q)

# Monitor GPU memory usage during execution
gpu_memory_allocated = torch.cuda.memory_allocated()

# Assert that GPU memory usage exceeds expected threshold indicating memory overflow
expected_threshold = 10 * 1024 * 1024 * 1024  # Example threshold in bytes
assert gpu_memory_allocated > expected_threshold