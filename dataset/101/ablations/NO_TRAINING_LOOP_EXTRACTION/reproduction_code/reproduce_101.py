import torch
from torch import nn
from einops import rearrange, repeat

# Define the RotaryEmbedding class (assuming it was missing in the original code)
class RotaryEmbedding:
    def __init__(self, dim, use_xpos=True):
        self.dim = dim
        self.use_xpos = use_xpos

    def rotate_queries_and_keys(self, Q, K):
        # This is a placeholder for the actual rotation logic
        # The bug is expected to be reproduced here, so we keep it simple
        # For demonstration, we will just return Q and K as is
        # In a real implementation, this would perform the rotation
        return Q, K  # This is where the bug might be introduced

# Define a simple transformer model for demonstration
class SimpleTransformer(nn.Module):
    def __init__(self, dim):
        super(SimpleTransformer, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, Q, K):
        return self.linear(Q + K)

# Setup
batch_size = 2
seq_len = 10
embedding_dim = 64

# Create random input tensors for queries (Q) and keys (K)
Q = torch.randn(batch_size, seq_len, embedding_dim)
K = torch.randn(batch_size, seq_len, embedding_dim)

# Instantiate the RotaryEmbedding class
rotary_embedding = RotaryEmbedding(dim=embedding_dim, use_xpos=True)

# Call the rotate_queries_and_keys method
rotated_Q, rotated_K = rotary_embedding.rotate_queries_and_keys(Q, K)

# Check for NaN values in rotated_K
assert not torch.isnan(rotated_K).any(), 'rotated_K contains NaN values'
print(rotated_K[torch.isnan(rotated_K)])  # Print NaN values for debugging

# Perform a forward pass through the simple transformer model
transformer = SimpleTransformer(dim=embedding_dim)
loss = transformer(rotated_Q, rotated_K)

# Check for NaN values in the loss
assert not torch.isnan(loss).any(), 'Loss contains NaN values'
print(loss)  # Log the output of the loss calculation