import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.cos_cached = torch.randn(10, 1, 1, dim)
        self.sin_cached = torch.randn(10, 1, 1, dim)

    def forward(self, x):
        x_rope = x
        neg_half_x = -x / 2
        # Bug: mismatched tensor dimensions
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

# Define the model and input
model = RotaryEmbedding(dim=4)
x = torch.randn(3, 1, 1, 3)  # Input with 3 features in the last dimension

# Run the model to reproduce the error
try:
    output = model(x)
    print(output)
except RuntimeError as e:
    print(f"Error: {e}")