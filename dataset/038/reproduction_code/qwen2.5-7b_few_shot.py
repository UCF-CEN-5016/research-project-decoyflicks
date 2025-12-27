import torch

class RotaryEmbedding:
    def __init__(self, batch_size=2, seq_len=5, hidden_dim=4):
        self.cos_cached = torch.randn(batch_size, seq_len, hidden_dim)
        self.sin_cached = torch.randn(batch_size, seq_len, hidden_dim)

    def apply_rotary(self, x):
        batch_size, seq_len, _ = x.shape
        neg_half_x = torch.zeros_like(x)  # Placeholder for actual computation
        x_rope = (x * self.cos_cached[:batch_size]) + (neg_half_x * self.sin_cached[:batch_size])
        return x_rope

# Test case
model = RotaryEmbedding()
x = torch.randn(2, 5, 3)  # shape (batch, seq, 3)
try:
    result = model.apply_rotary(x)
    print(result.shape)
except RuntimeError as e:
    print(f"Error: {e}")