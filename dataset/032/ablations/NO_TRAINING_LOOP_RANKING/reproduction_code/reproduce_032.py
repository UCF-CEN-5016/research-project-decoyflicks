import torch
import torch.nn.functional as F

def gelu(x):
    return F.gelu(x, approximate=True)

x = torch.randn(10, 768)

try:
    gelu(x)
except TypeError as e:
    print(e)

print("PyTorch version:", torch.__version__)