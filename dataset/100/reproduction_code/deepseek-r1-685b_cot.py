import torch
from torch import nn, einsum
from einops import repeat
from torch.cuda.amp import autocast

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        self.cached_freqs = None
        
    @autocast(enabled=False)
    def forward(self, t, seq_len=None, offset=0):
        # Cached version that causes the bug
        if seq_len is not None and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)]
            
        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        
        if seq_len is not None:
            self.cached_freqs = freqs
            
        return freqs

# Minimal training setup to reproduce
def reproduce_bug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Linear(32, 32).to(device)
    rotary = RotaryEmbedding(32).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # First iteration works fine
    x = torch.randn(1, 16, 32).to(device)
    t = torch.arange(16).float().to(device)
    freqs = rotary(t, seq_len=16)
    x_rot = x * freqs
    out = model(x_rot)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Second iteration fails
    x = torch.randn(1, 16, 32).to(device)
    t = torch.arange(16).float().to(device)
    freqs = rotary(t, seq_len=16)  # Tries to use cached freqs
    x_rot = x * freqs
    out = model(x_rot)
    loss = out.sum()
    loss.backward()  # RuntimeError here

if __name__ == '__main__':
    reproduce_bug()

if seq_len is not None:
    self.cached_freqs = freqs.detach()  # Detach from computation graph