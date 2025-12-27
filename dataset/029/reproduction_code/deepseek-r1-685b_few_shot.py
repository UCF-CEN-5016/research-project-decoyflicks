import torch

# Attempt to check for MPS backend (Metal Performance Shaders)
# This fails on PyTorch versions < 1.12 or non-macOS systems
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except AttributeError as e:
    print(f"Error: {e}")
    print("MPS backend not available in this PyTorch version")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple model to demonstrate device usage
model = torch.nn.Linear(10, 5).to(device)
x = torch.randn(3, 10).to(device)
print(model(x))