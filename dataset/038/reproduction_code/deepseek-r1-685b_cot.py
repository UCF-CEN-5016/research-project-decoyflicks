import torch
import torch.nn as nn

class RotaryEmbeddingBug(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.d = dim // 2  # Only apply to first d features
        self.max_seq_len = max_seq_len
        
        # Create full dimension cached values
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x):
        # x shape: (batch, seq_len, n_heads, head_dim)
        x_rope = x[..., :self.d]
        neg_half_x = torch.cat([-x_rope[..., self.d//2:], x_rope[..., :self.d//2]], dim=-1)
        
        # Buggy version - tries to multiply full dim cached with partial dim x
        # x_rope = (x_rope * self.cos_cached[:x.shape[1]]) + (neg_half_x * self.sin_cached[:x.shape[1]])
        
        # Fixed version - slices cached values to match x_rope dim
        x_rope = ((x_rope * self.cos_cached[:x.shape[1], :, :, :self.d]) +
                 (neg_half_x * self.sin_cached[:x.shape[1], :, :, :self.d]))
        
        return torch.cat([x_rope, x[..., self.d:]], dim=-1)

# Test case that triggers the bug
batch, seq_len, n_heads, head_dim = 2, 3, 4, 4
x = torch.randn(batch, seq_len, n_heads, head_dim)
rotary = RotaryEmbeddingBug(dim=head_dim)

# This would fail with original bug, works with fix
output = rotary(x)
print(output.shape)  # Should be same as input