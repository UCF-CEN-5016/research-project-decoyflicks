import torch
from torch import nn, einsum, Tensor

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = torch.randn(100, 10)  # Frequency tensor
        self.cached_freqs = None

    def forward(self, t: Tensor, seq_len: int = None, offset: int = 0):
        should_cache = seq_len is not None
        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)]

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = freqs.unsqueeze(1).repeat(1, 2, 1)  # Simulate repeat with 'r' dimension

        if should_cache:
            self.cached_freqs = freqs

        return freqs

# Initialize model and input
model = RotaryEmbedding()
t = torch.randn(1, 10)

# Training loop that produces error
for epoch in range(2):
    outputs = model(t)
    loss = torch.nn.functional.mse_loss(outputs, torch.randn_like(outputs))
    loss.backward()
    print(f"Epoch {epoch}: Loss = {loss.item()}")