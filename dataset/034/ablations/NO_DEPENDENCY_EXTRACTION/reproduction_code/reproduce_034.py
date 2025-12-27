import torch
import torch.nn.functional as F

def gelu(x, approximate):
    return F.gelu(x, approximate=True)

x = torch.randn(12, 768)
output = gelu(x, approximate=True)
print(output.shape)

# Triggering the TypeError
output = gelu(x, approximate=False)  # This will raise the TypeError