import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d, 2).float() / self.d))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        if self.cos_cached is None or self.cos_cached.shape[0] < x.shape[0]:
            t = torch.arange(x.shape[0], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

class TestModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim)
        self.d = dim // 2

    def forward(self, x):
        cos, sin = self.rotary_emb(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = torch.cat([-x_rope[..., self.d//2:], x_rope[..., :self.d//2]], dim=-1)
        x_rope = (x_rope * cos[:x.shape[0]]) + (neg_half_x * sin[:x.shape[0]])
        return torch.cat((x_rope, x_pass), dim=-1)

model = TestModel(dim=4)
x = torch.randn(3, 1, 1, 3)
output = model(x)