import torch
from torch.nn.functional import gelu

model = torch.nn.Sequential(
    torch.nn.Linear(10, 50),
    torch.nn.GELU(approximate=True)
)

try:
    X = torch.randn(32, 10)
    model(X)
except TypeError as e:
    print(f"TypeError occurred: {e}")