import torch
from torch import nn
from torch.cuda.amp import autocast

class RotaryEmbedding(nn.Module):
    def __init__(self, num_freqs=32):
        super(RotaryEmbedding, self).__init__()
        self.freqs = nn.Parameter(torch.randn(num_freqs))  # Random frequencies
        self.cached_freqs = None
        
    def compute_rotary_embeddings(self, t):
        freqs = self.freqs.unsqueeze(0)  # Add batch dimension
        freqs = torch.matmul(t.type_as(freqs), freqs)
        freqs = freqs.unsqueeze(-1).repeat(1, 1, 2)
        self.cached_freqs = freqs  # Cache the computed frequencies
        return freqs

    @autocast(enabled=False)
    def forward(self, t, seq_len=None, offset=0):
        if self.cached_freqs is not None:
            return self.cached_freqs[:, offset:(offset + seq_len)]
        return self.compute_rotary_embeddings(t)

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