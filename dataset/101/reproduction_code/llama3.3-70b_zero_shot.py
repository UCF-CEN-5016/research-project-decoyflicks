import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, x, seq_len):
        freqs = torch.arange(0, self.max_freq, dtype=torch.float32)
        freqs = 1 / (10000 ** (freqs / self.dim))
        t = torch.arange(seq_len, dtype=torch.float32)[:, None]
        phase = t * freqs
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

class RotaryEmbeddingPositional(nn.Module):
    def __init__(self, max_freq=10):
        super().__init__()
        self.max_freq = max_freq

    def forward(self, seq_len):
        freqs = torch.arange(0, self.max_freq, dtype=torch.float32)
        freqs = 1 / (10000 ** (freqs / 128))
        t = torch.arange(seq_len, dtype=torch.float32)[:, None]
        phase = t * freqs
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

class RotaryEmbeddingPositional2(nn.Module):
    def __init__(self, max_freq=10):
        super().__init__()
        self.max_freq = max_freq

    def forward(self, seq_len):
        freqs = torch.arange(0, self.max_freq, dtype=torch.float32)
        freqs = 1 / (10000 ** (freqs / 128))
        t = torch.arange(seq_len, dtype=torch.float32)[:, None]
        phase = t * freqs
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

def rotate_queries_and_keys(Q, K, xpos):
    embed = RotaryEmbeddingPositional2(max_freq=10)
    pos = embed(len(Q)).to(Q.device)
    pos = pos[:, None, :]
    Q = Q + pos
    K = K + pos
    return Q, K

Q = torch.randn(10, 10, 128)
K = torch.randn(10, 10, 128)
xpos = torch.randn(10, 10)
Q, K = rotate_queries_and_keys(Q, K, xpos)
print(torch.isnan(K).sum())