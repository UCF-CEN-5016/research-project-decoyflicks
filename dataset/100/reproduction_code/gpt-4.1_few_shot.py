import torch
from torch import einsum, Tensor
from torch.nn import Module
from torch.cuda.amp import autocast

def exists(val):
    return val is not None

def repeat(tensor, pattern, **kwargs):
    # Simplified repeat implementation using torch.tile for demonstration
    # pattern '... n -> ... (n r)' means repeat last dim r times
    r = kwargs.get('r', 1)
    shape = list(tensor.shape)
    shape[-1] *= r
    return tensor.repeat_interleave(r, dim=-1)

class RotaryEmbedding(Module):
    def __init__(self):
        super().__init__()
        # Precompute frequencies tensor (dummy example)
        self.freqs = torch.linspace(0, 1, steps=16)
        self.cached_freqs = None
        self.learned_freq = False
        self.freqs_for = 'token'

    def tmp_store(self, attr, value):
        setattr(self, attr, value)

    @autocast(enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel'
        )

        if (
            should_cache and
            exists(self.cached_freqs) and
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            # Returns cached tensor that keeps reference to computation graph
            return self.cached_freqs[offset:(offset + seq_len)]

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache:
            # Cache tensor with attached computation graph, causing error on 2nd backward
            self.tmp_store('cached_freqs', freqs)

        return freqs

# Dummy training loop to reproduce error
model = RotaryEmbedding()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Input tensor requires grad for backward
t = torch.randn(8, requires_grad=True)

for i in range(2):
    optimizer.zero_grad()
    freqs = model(t, seq_len=t.shape[0], offset=0)
    loss = freqs.sum()
    loss.backward()  # RuntimeError on second iteration due to cached graph
    optimizer.step()
    print(f"Iteration {i} completed")