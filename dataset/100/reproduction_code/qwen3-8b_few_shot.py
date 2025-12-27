import torch

class CacheLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cached = None

    def forward(self, x):
        if self.cached is None:
            self.cceived = x * 2  # Compute and cache
        return self.cached

model = CacheLayer()
x = torch.randn(10, requires_grad=True)
loss = model(x).sum()
loss.backward()  # First backward
loss.backward()  # Second backward, causes error

import torch

class CacheLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cached = None

    def forward(self, x):
        if self.cached is None:
            # Compute and cache the tensor
            self.cached = x * 2
        return self.cached

model = CacheLayer()
x = torch.randn(10, requires_grad=True)
loss = model(x).sum()

# First backward pass
loss.backward()

# Second backward pass (causes error)
loss.backward()