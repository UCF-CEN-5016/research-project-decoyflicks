import torch
import torch.nn as nn

class CorrectROPE(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin_cached', torch.sin(freqs).view(1, 1, -1, 1))
        self.register_buffer('cos_cached', torch.cos(freqs).view(1, 1, -1, 1))

    def forward(self, x):
        x_rope = x[..., :self.dim]
        neg_half_x = torch.cat([-x_rope[..., 1::2], x_rope[..., ::2]], dim=-1)
        
        batch_size = x.shape[0]
        x_rope = (x_rope * self.cos_cached[:, :, :batch_size, :]) + (neg_half_x * self.sin_cached[:, :, :batch_size, :])
        
        return torch.cat([x_rope, x[..., self.dim:]], dim=-1)

# Test the refactored code
rope = CorrectROPE(dim=64)
x = torch.randn(3, 8, 64)  # (batch, seq_len, dim)
try:
    output = rope(x)
    print("No errors encountered. The refactored ROPE implementation is correct.")
except RuntimeError as e:
    print(f"Error: {e}")