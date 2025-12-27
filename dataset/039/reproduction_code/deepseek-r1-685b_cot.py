import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10000):
        super().__init__()
        self.d = d
        self.base = base
        self.cos_cached = None
        self.sin_cached = None
    
    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[None, None, :, :]
        self.sin_cached = idx_theta2.sin()[None, None, :, :]
    
    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        
        # Split features into pairs
        x_rope = x[..., :self.d]
        x_pass = x[..., self.d:]
        
        # Negate half of x_rope for rotation
        neg_half_x = torch.cat([-x_rope[..., 1::2], x_rope[..., ::2]], dim=-1)
        
        # Original buggy line:
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        
        # Fixed version would be:
        # x_rope = (x_rope * self.cos_cached[:, :, :x.shape[0], :]) + (neg_half_x * self.sin_cached[:, :, :x.shape[0], :])
        
        return torch.cat((x_rope, x_pass), dim=-1)

# Trigger the bug
d_model = 64
seq_len = 4
batch_size = 2
heads = 3

x = torch.randn(seq_len, batch_size, heads, d_model)
rope = RotaryPositionalEmbeddings(d=d_model//2)  # Using half features for RoPE
try:
    out = rope(x)  # This will trigger the dimension mismatch error
except RuntimeError as e:
    print("Error reproduced:")
    print(e)