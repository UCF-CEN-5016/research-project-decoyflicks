import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.cos_cached = torch.randn(1, 1, dim, max_seq_len)
        self.sin_cached = torch.randn(1, 1, dim, max_seq_len)

    def forward(self, x):
        # Simulate incorrect implementation
        neg_half_x = x[:, :, :, 1::2]
        x_rope = x[:, :, :, ::2]
        
        # This line will cause the error
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        
        return x_rope

# Create a sample input
x = torch.randn(3, 1, 10, 8)

# Initialize the model
model = RotaryPositionalEmbeddings(dim=10, max_seq_len=8)

# Run the model to reproduce the error
try:
    output = model(x)
    print(output)
except RuntimeError as e:
    print(e)