import torch

class CacheLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cached = None

    def compute_and_cache(self, x):
        self.cached = x * 2

    def forward(self, x):
        if self.cached is None:
            self.compute_and_cache(x)
        return self.cached

model = CacheLayer()
x = torch.randn(10, requires_grad=True)
loss = model(x).sum()

# First backward pass
loss.backward()

# Second backward pass (causes error)
loss.backward()