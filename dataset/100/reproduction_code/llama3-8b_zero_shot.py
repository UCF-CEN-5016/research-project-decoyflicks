import torch
from torch import Tensor

class RotaryEmbedding:
    def __init__(self):
        self.learned_freq = False
        self.cached_freqs = None

    @torch.autocast(enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        if not self.learned_freq and exists(seq_len) and self.freqs_for != 'pixel':
            return self.cached_freqs[offset:(offset + seq_len)]

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat(freqs, '... n -> ... (n r)', r=2)

        return freqs

    def tmp_store(self, key: str, value):
        setattr(self, key, value)

class Trainer:
    def __init__(self, rotary_embedding):
        self.rotary_embedding = rotary_embedding
        self.cached_freqs = None

    def train(self):
        t = torch.randn(1, 10)
        for _ in range(2):
            seq_len = 5
            offset = _ % 2
            freqs = self.rotary_embedding.forward(t, seq_len, offset)

            if _ == 0:
                self.cached_freqs = freqs

            freqs.backward()

trainer = Trainer(RotaryEmbedding())
trainer.train()