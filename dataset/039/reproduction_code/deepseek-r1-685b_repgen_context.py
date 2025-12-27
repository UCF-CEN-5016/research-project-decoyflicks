import torch
import torch.nn as nn

class CorrectROPE(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin_cached', torch.sin(freqs).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cos_cached', torch.cos(freqs).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x_rope = x[..., :self.dim]
        neg_half_x = torch.cat([-x_rope[..., 1::2], x_rope[..., ::2]], dim=-1)

        batch_size = x.size(0)
        cos_slice = self.cos_cached[:, :, :x_rope.size(1)]
        sin_slice = self.sin_cached[:, :, :x_rope.size(1)]
        
        x_rope = (x_rope.unsqueeze(2) * cos_slice) + (neg_half_x.unsqueeze(2) * sin_slice)
        
        return torch.cat([x_rope.squeeze(2), x[..., self.dim:]], dim=-1)

# Test case that triggers the bug
rope = CorrectROPE(dim=64)
x = torch.randn(3, 8, 64)  # (batch, seq_len, dim)
try:
    output = rope(x)  # Will raise RuntimeError
except RuntimeError as e:
    print(f"Error: {e}")
    print("The bug occurs due to incorrect tensor slicing in the ROPE implementation")