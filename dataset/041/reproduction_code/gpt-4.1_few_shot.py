import torch
import torch.nn as nn

# Simple model without 'mask' argument in forward
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(2, 10)
mask = torch.ones(2, 10)

# This call raises TypeError because forward() has no 'mask' parameter
output = model(x, mask=mask)