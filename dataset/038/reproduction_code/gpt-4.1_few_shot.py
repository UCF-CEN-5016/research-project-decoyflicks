import torch

class RotaryEmbedding:
    def __init__(self, seq_len, dim):
        # Assume dim is total feature dim used for cos/sin cache
        self.d = dim
        # Cache cos and sin with shape (seq_len, 1, 1, dim)
        self.cos_cached = torch.randn(seq_len, 1, 1, dim)
        self.sin_cached = torch.randn(seq_len, 1, 1, dim)

    def apply_rotary(self, x):
        # x shape: (seq_len, batch_size, num_heads, partial_dim)
        # partial_dim < self.d causes dimension mismatch
        x_rope = x.clone()
        neg_half_x = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)  # some transform

        # Buggy line - will error if x.shape[-1] != self.d
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

# Setup
seq_len = 4
batch_size = 2
num_heads = 1
full_dim = 4  # dim of cached cos/sin
partial_dim = 3  # smaller partial feature dim, causes mismatch

rotary = RotaryEmbedding(seq_len, full_dim)

# Input with smaller feature dimension than cos/sin caches
x = torch.randn(seq_len, batch_size, num_heads, partial_dim)

# This triggers the RuntimeError due to size mismatch
output = rotary.apply_rotary(x)
print(output)