import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim // 2  # Only applying to half the features
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, x):
        # Initialize cached sin/cos if not done
        if self.cos_cached is None or self.cos_cached.shape[0] < x.shape[0]:
            self._build_cache(x)
            
        # Split features for rotary embedding
        x_rope = x[..., :self.d]
        neg_half_x = torch.cat([-x_rope[..., self.d//2:], x_rope[..., :self.d//2]], dim=-1)
        
        # Original buggy implementation
        # x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        
        # Corrected implementation
        x_rope = ((x_rope * self.cos_cached[:x.shape[0], :, :, :self.d]) +
                 (neg_half_x * self.sin_cached[:x.shape[0], :, :, :self.d]))
        
        return torch.cat([x_rope, x[..., self.d:]], dim=-1)
    
    def _build_cache(self, x):
        # Dummy cache building (simplified for reproduction)
        seq_len = x.shape[0]
        self.cos_cached = torch.ones(seq_len, 1, 1, x.shape[-1])  # Full feature dim
        self.sin_cached = torch.ones(seq_len, 1, 1, x.shape[-1])  # Full feature dim

# Test case
embed = RotaryEmbedding(dim=4)  # Will only apply to first 2 features
x = torch.randn(3, 1, 1, 4)    # Batch of 3, 4 features
output = embed(x)               # Should work with corrected implementation
print(output.shape)