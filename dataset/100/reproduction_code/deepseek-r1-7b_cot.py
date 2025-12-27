import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super(RotaryEmbedding, self).__init__()
        self.register_buffer('cache', None)
        self.freqs = None
        
    @torch.no_grad()
    def forward(self, t, seq_len=None, offset=0):
        if self.cache is not None and exists(seq_len) and (offset + seq_len) <= self.cache.shape[0]:
            return self.cache[offset:(offset + seq_len)]
        
        freqs = self.freqs
        freqs = einsum('..., f -> ...f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '...n -> ... (n r)', r=2)
        
        if self.cache is not None and exists(seq_len) and (offset + seq_len) <= self.cache.shape[0]:
            self.cache[offset:(offset + seq_len)] = freqs
        
        return freqs

# Usage
model = RotaryEmbedding()
loss = model(some_tensor)
loss.backward()

# Second backward pass without retain_graph, causing error
try:
    loss.backward()
except RuntimeError as e:
    print(f"RuntimeError: {e}")

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super(RotaryEmbedding, self).__init__()
        self.register_buffer('cache', None)
        
    @torch.no_grad()
    def forward(self, t, seq_len=None, offset=0):
        if self.cache is not None and exists(seq_len) and (offset + seq_len) <= self.cache.shape[0]:
            return self.cache[offset:(offset + seq_len)]
        
        freqs = einsum('..., f -> ...f', t.type(self.freqs.dtype), self.freqs)
        freqs = repeat(freqs, '...n -> ... (n r)', r=2)
        
        if self.cache is not None and exists(seq_len) and (offset + seq_len) <= self.cache.shape[0]:
            self.cache[offset:(offset + seq_len)] = freqs
        
        return freqs

    @torch.no_grad()
    def forwardWithCache(self, t, seq_len=None, offset=0):
        if not hasattr(self, 'cache'):  # Initialize on first use
            self.cache = torch.zeros(...)
        
        return super().forward(t, seq_len, offset)

# Usage with retain_graph=True when needed
model = RotaryEmbedding()
loss = model(some_tensor)
loss.backward(retain_graph=True)  # Retains intermediates for multiple passes

try:
    loss.backward()  # Now works without error as intermediates are retained
except RuntimeError as e:
    print(f"RuntimeError: {e}")