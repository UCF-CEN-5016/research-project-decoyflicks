import torch
from rotary_embedding_torch import RotaryEmbedding

# Define input dimensions
dim = 512
batch_size = 64
seq_len = 10

# Generate random Q and K tensors
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)

# Initialize RotaryEmbedding instance with use_xpos set to True
rotary_embedding = RotaryEmbedding(dim, use_xpos=True)

# Apply rotation on Q and K
rotated_Q, rotated_K = rotary_embedding.rotate_queries_and_keys(Q, K)

# Calculate the percentage of NaN values in rotated_K
nan_percentage = torch.isnan(rotated_K).sum().item() / (batch_size * seq_len * dim) * 100

# Print the percentage of NaN values
print(f"Percentage of NaN values in rotated_K: {nan_percentage:.2f}%")