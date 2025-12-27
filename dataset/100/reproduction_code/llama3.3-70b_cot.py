import torch
from torch import nn, autocast
from transformers import Accelerator

# Set up minimal environment
accelerator = Accelerator()

# Define a simple rotary embedding layer
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.freqs = nn.Parameter(torch.randn(dim))
        self.cached_freqs = None

    @autocast(enabled=False)
    def forward(self, t, seq_len=None, offset=0):
        should_cache = not self.learned_freq and seq_len is not None
        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)]

        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        if should_cache:
            self.cached_freqs = freqs

        return freqs

# Define a simple neural network with a rotary embedding layer
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_embedding = RotaryEmbedding(dim=256, max_seq_len=512)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.rotary_embedding(x)
        x = self.fc(x)
        return x

# Initialize the network and optimizer
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Wrap the network and optimizer with Hugging Face Accelerate
net, optimizer = accelerator.prepare(net, optimizer)

# Define a simple training loop
for epoch in range(2):
    # Generate some random input data
    x = torch.randn(1, 256)

    # Forward pass
    outputs = net(x)
    loss = outputs.sum()

    # Backward pass
    accelerator.backward(loss)

    # Update the network parameters
    optimizer.step()
    optimizer.zero_grad()