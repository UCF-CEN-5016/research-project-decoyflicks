import torch
from torch import nn, autocast, einsum
from torch.nn.functional import mse_loss

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.linspace(1.0, 10.0, 16), requires_grad=False)
        self.cached_freqs = None

    @autocast(enabled=False)
    def forward(self, t, seq_len=None, offset=0):
        should_cache = True
        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)]

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)

        if should_cache:
            self.cached_freqs = freqs

        return freqs

model = RotaryEmbedding()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(2):
    optimizer.zero_grad()
    x = torch.randn(4, 16, requires_grad=True)
    out = model(x, seq_len=16)
    loss = mse_loss(out, torch.zeros_like(out))
    loss.backward()
    optimizer.step()