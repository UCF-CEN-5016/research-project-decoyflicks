import torch
import torch.nn as nn

class IncorrectROPE(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin_cached', torch.sin(freqs).view(1, 1, -1, 1))
        self.register_buffer('cos_cached', torch.cos(freqs).view(1, 1, -1, 1))

    def forward(self, x):
        # Problematic implementation
        x_rope = x[..., :self.dim]
        neg_half_x = torch.cat([-x_rope[..., 1::2], x_rope[..., ::2]], dim=-1)
        
        # This line causes the dimension mismatch error
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        
        return torch.cat([x_rope, x[..., self.dim:]], dim=-1)

# Test case that triggers the bug
rope = IncorrectROPE(dim=64)
x = torch.randn(3, 8, 64)  # (batch, seq_len, dim)
try:
    output = rope(x)  # Will raise RuntimeError
except RuntimeError as e:
    print(f"Error: {e}")
    print("The bug occurs due to incorrect tensor slicing in the ROPE implementation")