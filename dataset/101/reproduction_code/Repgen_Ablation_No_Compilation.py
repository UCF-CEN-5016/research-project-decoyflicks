import torch
import torch.nn as nn

# Define the RotaryEmbedding class assuming it's missing from the original code
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, use_xpos: bool = False, xpos_scale_base: int = 512):
        super().__init__()
        self.dim = dim
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base

    def rotate_queries_and_keys(self, Q: torch.Tensor, K: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Placeholder implementation for demonstration purposes
        # In practice, this method should implement the rotary embedding logic
        return Q.clone(), K.clone()

# Initialize the input query tensor Q and key tensor K
dim = 512
seq_len = 64
batch_size = 32

Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)

# Create an instance of RotaryEmbedding
rotary_embedding = RotaryEmbedding(dim=dim, use_xpos=True, xpos_scale_base=512)

# Rotate both Q and K tensors using the rotary embedding function
Q_rotated, K_rotated = rotary_embedding.rotate_queries_and_keys(Q), rotary_embedding.rotate_queries_and_keys(K)

# Print the number of NaN values in the rotated key tensor K
print(K_rotated.isnan().sum() / (batch_size * seq_len))