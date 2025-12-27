

import torch

class RotaryEmbedding:
    def __init__(self):
        self.cos_cached = torch.randn(2, 5, 4)  # batch, seq, 4
        self.sin_cached = torch.randn(2, 5, 4)

    def apply_rotary(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x_rope = x  # Assume x has shape (batch, seq, 3)
        neg_half_x = x  # some operation
        x_rope = (x_rope * self.cos_cached[:batch_size]) + (neg_half_x * self.sin_cached[:batch_size])
        return x_rope

# Test case
model = RotaryEmbedding()
x = torch.randn(2, 5, 3)  # shape (batch, seq, 3)
result = model.apply_rotary(x)
print(result.shape)

import torch

class RotaryEmbedding:
    def __init__(self):
        self.cos_cached = torch.randn(2, 5, 4)  # batch, seq, 4
        self.sin_cached = torch.randn(2, 5, 4)

    def apply_rotary(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x_rope = x  # Assume x has shape (batch, seq, 3)
        neg_half_x = x  # Placeholder for actual computation
        x_rope = (x_rope * self.cos_cached[:batch_size]) + (neg_half_x * self.sin_cached[:batch_size])
        return x_rope

# Test case
model = RotaryEmbedding()
x = torch.randn(2, 5, 3)  # shape (batch, seq, 3)
try:
    result = model.apply_rotary(x)
    print(result.shape)
except RuntimeError as e:
    print(f"Error: {e}")