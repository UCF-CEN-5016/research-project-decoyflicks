import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Assume x has shape (batch, seq_len, dim)
        # Compute attention using einsum: 'b i d, b j d -> b i j'
        attention = torch.einsum('b i d, b j d -> b i j', x, x)
        # Incorrect softmax: applied along dim=2 (third dimension) instead of dim=1 (second)
        attn = attention.softmax(dim=2)  # Bug: should be dim=1
        return attn

# Test with a sample input
model = SelfAttention(dim=64)
input_tensor = torch.randn(1, 8, 64)  # batch=1, seq_len=8, dim=64
output = model(input_tensor)
print("Attention output shape:", output.shape)
print("Attention values:\n", output)