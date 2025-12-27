import torch
from transformers import RotaryEmbedding

# Create a model with rotary embedding without caching
class RotaryEmbeddingModel(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self rotary_embedding = RotaryEmbedding(dim=dim)
        
    def forward(self, x, seq_len=None, offset=0):
        # Compute frequency on the fly without caching
        freqs = torch.randn(10, 50)  # Simulating cached_freqs computation
        
        return freqs

# Initialize model and data
model = RotokayEmbeddingModel()
x = torch.randn(32, 10)

# Training loop (this will not cause the RuntimeError)
for _ in range(100):
    optimizer = ...  # Initialization code
    outputs = model(x)
    loss = ...
    
    optimizer.zero_grad()
    loss.backward()

import torch

class MinimalModel(torch.nn.Module):
    def forward(self, x):
        # Compute output without caching
        return x * (torch.randn(10) + 1).view(-1, 1)

# Initialize model and data
model = MinimalModel()
x = torch.randn(32, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for _ in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = (outputs * outputs).sum()
    loss.backward()
    optimizer.step()