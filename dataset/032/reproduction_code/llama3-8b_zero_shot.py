import torch.nn.functional as F

def gelu(x):
    return F.gelu(x, approximate=True)

print(gelu(torch.tensor([1., 2.])))