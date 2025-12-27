import torch
from torch import nn, einsum
from einops import repeat
from torch.cuda.amp import autocast

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = torch.randn(32)  # Random frequencies
        self.cached_freqs = None
        
    @autocast(enabled=False)
    def forward(self, t, seq_len=None, offset=0):
        # With caching (produces error on second backward)
        if self.cached_freqs is not None:
            return self.cached_freqs[offset:(offset + seq_len)]
            
        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        self.cached_freqs = freqs  # Cache the computed frequencies
        return freqs

# Test setup
model = RotaryEmbedding()
optimizer = torch.optim.Adam(model.parameters())
x = torch.randn(1, 32, requires_grad=True)

# First backward pass works
out1 = model(x, seq_len=32)
loss1 = out1.sum()
loss1.backward()
optimizer.step()

# Second backward pass fails
out2 = model(x, seq_len=32)
loss2 = out2.sum()
loss2.backward()  # RuntimeError here
optimizer.step()