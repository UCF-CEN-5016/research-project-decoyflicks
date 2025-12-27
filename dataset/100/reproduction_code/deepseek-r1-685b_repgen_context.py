import torch
from torch import nn, einsum
from einops import repeat

class RotaryEmbedding(nn.Module):
    def __init__(self, num_freqs=32):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_freqs))  # Random frequencies
        self.cached_freqs = None
        
    def forward(self, t, seq_len=None, offset=0):
        if self.cached_freqs is not None:
            return self.cached_freqs[offset:(offset + seq_len)]
            
        freqs = einsum('...f, f -> ...f', t.type(self.freqs.dtype), self.freqs)
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

# Second backward pass works
out2 = model(x, seq_len=32)
loss2 = out2.sum()
loss2.backward()
optimizer.step()