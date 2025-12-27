import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim // 2  # Only applying to half the features
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        self._build_cache(x)
        
        x_rope = x[..., :self.d]
        neg_half_x = torch.cat([x_rope[..., self.d//2:], x_rope[..., :self.d//2]], dim=-1)
        
        x_rope = ((x_rope * self.cos_cached) + (neg_half_x * self.sin_cached))
        
        return torch.cat([x_rope, x[..., self.d:]], dim=-1)

    def _build_cache(self, x):
        seq_len = x.shape[0]
        if self.cos_cached is None or self.cos_cached.shape[0] < seq_len:
            self.cos_cached = torch.ones(seq_len, 1, 1, self.d)
            self.sin_cached = torch.ones(seq_len, 1, 1, self.d)

# Test case
embed = RotaryEmbedding(dim=4)  # Will only apply to first 2 features
x = torch.randn(3, 1, 1, 4)    # Batch of 3, 4 features
output = embed(x)               # Should work with corrected implementation
print(output.shape)