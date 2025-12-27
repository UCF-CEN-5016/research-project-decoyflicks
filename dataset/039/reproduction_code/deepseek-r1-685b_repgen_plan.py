import torch
import torch.nn as nn

class ROPE(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin_cached', torch.sin(freqs).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cos_cached', torch.cos(freqs).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x_rope = x[..., :self.dim]
        neg_half_x = torch.cat([-x_rope[..., 1::2], x_rope[..., ::2]], dim=-1)
        
        batch_size = x.size(0)
        cos_slice = self.cos_cached[:, :, :batch_size, :]
        sin_slice = self.sin_cached[:, :, :batch_size, :]
        
        x_rope = (x_rope.unsqueeze(2) * cos_slice) + (neg_half_x.unsqueeze(2) * sin_slice)
        
        return torch.cat([x_rope.squeeze(2), x[..., self.dim:]], dim=-1)

# Test case with the refactored ROPE module
rope = ROPE(dim=64)
x = torch.randn(3, 8, 64)  # (batch, seq_len, dim)
output = rope(x)
print("Output shape:", output.shape)