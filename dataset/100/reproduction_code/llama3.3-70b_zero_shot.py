import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.cached_freqs = None

    def forward(self, t, seq_len=None, offset=0):
        freqs = self.get_freqs()
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = freqs.repeat(1, 2)

        if seq_len is not None and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            self.cached_freqs = freqs
            return self.cached_freqs[offset:(offset + seq_len)]

        return freqs

    def get_freqs(self):
        if not hasattr(self, 'freqs'):
            freqs = torch.arange(self.max_freq, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.freqs = 1 / (10000 ** (freqs / self.dim))
        return self.freqs

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_embedding = RotaryEmbedding(dim=16)

    def forward(self, t, seq_len=None, offset=0):
        return self.rotary_embedding(t, seq_len, offset)

model = Model()
t = torch.randn(1, 1)
loss = model(t).sum()
loss.backward()

t = torch.randn(1, 1)
loss = model(t).sum()
loss.backward()