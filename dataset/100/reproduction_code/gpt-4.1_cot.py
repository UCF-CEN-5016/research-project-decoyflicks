import torch
from torch import nn, Tensor

class CachedFreqModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate frequency parameter
        self.freqs = nn.Parameter(torch.randn(4))
        self.cached_freqs = None

    def forward(self, t: Tensor):
        # Cache computation, simulating the bug
        if self.cached_freqs is not None:
            # Return cached tensor without detach
            return self.cached_freqs
        freqs = self.freqs
        # Compute a tensor depending on input and freqs
        out = t * freqs
        self.cached_freqs = out  # cache without detach
        return out

# Setup
model = CachedFreqModule()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy inputs
x = torch.randn(2, 4, requires_grad=True)
y = torch.randn(2, 4)

# First iteration
optimizer.zero_grad()
out = model(x)
loss = ((out - y) ** 2).mean()
loss.backward()      # OK
optimizer.step()

# Second iteration
optimizer.zero_grad()
out = model(x)
loss = ((out - y) ** 2).mean()
loss.backward()      # This will raise RuntimeError
optimizer.step()