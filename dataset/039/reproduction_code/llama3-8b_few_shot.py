import torch

# Minimalistic example to reproduce the bug
class ROPEImplementation:
    def __init__(self):
        self.cos_cached = torch.randn(1, 10, 10)  # Initialize cosine cached values
        self.sin_cached = torch.randn(1, 10, 10)  # Initialize sine cached values

    def forward(self, x_rope):
        neg_half_x = -x_rope / 2.0
        x_rope = (x_rope * self.cos_cached[:x_rope.shape[0]]) + (neg_half_x * self.sin_cached[:x_rope.shape[0]])
        return x_rope

rope_impl = ROPEImplementation()
x_rope = torch.randn(3)  # Input to the buggy implementation
try:
    result = rope_impl.forward(x_rope)
except RuntimeError as e:
    print(f"Error: {e}")