import torch
from torch import nn, einsum, Tensor
from torch.cuda.amp import autocast
from einops import repeat
from typing import Optional

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = torch.randn(64)
        self.cached_freqs: Optional[Tensor] = None

    def tmp_store(self, name, value):
        setattr(self, name, value)

    @autocast(enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = seq_len is not None

        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)]

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache:
            self.tmp_store('cached_freqs', freqs)

        return freqs

model = RotaryEmbedding().cuda()
optimizer = torch.optim.Adam(model.parameters())

for _ in range(2):
    x = torch.randn(1, 10, requires_grad=True).cuda()
    out = model(x, seq_len=10)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()