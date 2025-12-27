import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} features, got {x.shape[-1]}")

x = torch.randn(1, 10, 4)
rotary_pe = RotaryPositionalEmbeddings(3)
try:
    rotary_pe(x)
except ValueError as e:
    print(e)