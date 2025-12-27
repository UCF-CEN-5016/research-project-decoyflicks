import torch
import torch.nn.functional as F

def gelu(x, approximate):
    return F.gelu(x, approximate=True)

if __name__ == '__main__':
    torch.manual_seed(42)
    input_tensor = torch.randn(12, 768)
    gelu(input_tensor, approximate=True)