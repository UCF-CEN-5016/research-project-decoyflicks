import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim // 2  # Only applying to half the features
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, x):
        if self.cos_cached is None or self.cos_cached.shape[0] < x.shape[0]:
            self._build_cache(x)
            
        x_rope = x[..., :self.d]
        neg_half_x = torch.cat([-x_rope[..., self.d//2:], x_rope[..., :self.d//2]], dim=-1)
        
        x_rope = ((x_rope * self.cos_cached[:x.shape[0]]) +
                 (neg_half_x * self.sin_cached[:x.shape[0]))
        
        return torch.cat([x_rope, x[..., self.d:]], dim=-1)
    
    def _build_cache(self, x):
        self.cos_cached = torch.cos(torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(1) / x.shape[0])
        self.sin_cached = torch.sin(torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(1) / x.shape[0])

# Test case
embed = RotaryEmbedding(dim=4)  # Will only apply to first 2 features
x = torch.randn(3, 1, 1, 4)    # Batch of 3, 4 features
output = embed(x)               # Should work with corrected implementation
print(output.shape)