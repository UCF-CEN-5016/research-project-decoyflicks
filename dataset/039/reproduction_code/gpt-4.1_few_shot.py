import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create cached cos and sin tensors of shape (max_seq_len, dim)
        # For simplicity, we simulate them as random tensors here
        self.cos_cached = torch.randn(max_seq_len, dim)
        self.sin_cached = torch.randn(max_seq_len, dim)

    def forward(self, x):
        # x shape: (seq_len, batch_size, dim)
        # For demonstration, assume seq_len=3, batch_size=2, dim matches self.dim
        seq_len = x.shape[0]

        # Split x into two halves along dim dimension
        # Assuming dim is even
        half_dim = self.dim // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        
        # Rotate half of the features (simulated here)
        neg_half_x = torch.cat((-x2, x1), dim=-1)  # shape: (seq_len, batch_size, dim)

        # Attempt to apply rotary embeddings with incorrect slicing
        # This causes RuntimeError when seq_len=3 but cached cos/sin have max_seq_len=4
        # because .cos_cached[:seq_len] has shape (3, dim) but x has shape starting with (3, batch_size, dim)
        # The mismatch is in dimension 1 or 0 during broadcasting
        # The error occurs here:
        x_rope = (x * self.cos_cached[:seq_len]) + (neg_half_x * self.sin_cached[:seq_len])

        return x_rope

# Sample input tensor with seq_len=3, batch_size=2, dim=6
x = torch.randn(3, 2, 6)

model = RotaryPositionalEmbedding(dim=6, max_seq_len=4)
output = model(x)  # This line raises RuntimeError due to shape mismatch

print(output)