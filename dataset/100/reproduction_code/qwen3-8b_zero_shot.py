import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([1.0, 2.0], requires_grad=True)
        self.cached_weight = None

    def forward(self, x):
        if self.cached_weight is None:
            self.cached_weight = self.weight.clone()  # This is a detached copy
        return self.cached_weight * x

# Test the model
model = Model()
x = torch.tensor([3.0, 4.0], requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()

self.cached_weight = self.weight  # This is fine, as it's a reference

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([1.0, 2.0], requires_grad=True)
        self.cached_weight = None

    def forward(self, x):
        if self.cached_weight is None:
            self.cached_weight = self.weight  # This is a reference, not a copy
        return self.cached_weight * x

# Test the model
model = Model()
x = torch.tensor([3.0, 4.0], requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()