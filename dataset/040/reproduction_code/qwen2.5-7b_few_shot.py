import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Assume x has shape (batch, seq_len, dim)
        # Compute attention using matrix multiplication and softmax
        attention = torch.matmul(x, x.transpose(1, 2)) / (self.dim ** 0.5)
        attn = torch.nn.functional.softmax(attention, dim=1)
        return attn

# Test with a sample input
model = SelfAttention(dim=64)
input_tensor = torch.randn(1, 8, 64)  # batch=1, seq_len=8, dim=64
output = model(input_tensor)
print("Attention output shape:", output.shape)
print("Attention values:\n", output)