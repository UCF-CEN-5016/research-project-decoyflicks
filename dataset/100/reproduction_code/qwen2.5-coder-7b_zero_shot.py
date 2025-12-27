import torch
from torch import Tensor
from torch import nn

def exists(value) -> bool:
    return value is not None

class RotaryEmbedding:
    def __init__(self, freq_size: int = 10):
        # Whether frequencies are learned or fixed
        self.learned_freq = False
        # Storage for cached frequency tensors (can be set externally)
        self.cached_freqs = None
        # descriptor used in conditional checks; kept for compatibility
        self.freqs_for = None
        # learnable frequency parameters (leaf tensor so backward works)
        self.freqs = nn.Parameter(torch.randn(freq_size), requires_grad=True)

    @torch.autocast(enabled=False)
    def forward(self, t: Tensor, seq_len: int = None, offset: int = 0) -> Tensor:
        # Use cached frequencies only if they exist and all conditions match
        if (
            not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != 'pixel'
            and self.cached_freqs is not None
        ):
            return self.cached_freqs[offset : (offset + seq_len)]

        # Elementwise multiply along the last dimension between input and freqs
        freqs = torch.einsum('...f,f->...f', t.type(self.freqs.dtype), self.freqs)
        # Repeat each element in the last dimension twice
        freqs = freqs.repeat_interleave(2, dim=-1)

        return freqs

    def store_attribute(self, key: str, value):
        setattr(self, key, value)


class Trainer:
    def __init__(self, rotary_embedding: RotaryEmbedding):
        self.rotary_embedding = rotary_embedding

    def train(self):
        input_tensor = torch.randn(1, 10)
        for step in range(2):
            seq_len = 5
            offset = step % 2
            freqs = self.rotary_embedding.forward(input_tensor, seq_len, offset)

            if step == 0:
                # store the computed freqs in the rotary embedding for caching
                self.rotary_embedding.cached_freqs = freqs

            freqs.backward()


if __name__ == "__main__":
    trainer = Trainer(RotaryEmbedding())
    trainer.train()