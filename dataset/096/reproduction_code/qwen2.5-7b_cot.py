import torch
import random

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(2, requires_grad=True)

if random.random() < 0.5:
    with torch.no_grad():
        y = model(x)  # Forward pass without gradient computation
        z = y * 2
        z.backward()  # Now this will work, since y is still in the computation graph