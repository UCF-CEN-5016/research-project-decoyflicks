import torch
import random

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x, t):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(2, requires_grad=True)
t = torch.randn(1)

if random.random() < 0.5:
    with torch.inference_mode():
        y = model(x, t)  # Forward pass under inference mode
        z = y * 2
        z.backward()  # Attempt to compute gradients on a tensor from inference mode

import torch
import random

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x, t):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(2, requires_grad=True)
t = torch.randn(1)

if random.random() < 0.5:
    with torch.no_grad():
        y = model(x, t)  # Forward pass without gradient computation
        z = y * 2
        z.backward()  # Now this will work, since y is still in the computation graph