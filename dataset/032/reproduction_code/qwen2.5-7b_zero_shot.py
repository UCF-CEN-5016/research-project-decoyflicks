import torch
import torch.nn.functional as F

x = torch.tensor([1.0])
result = F.gelu(x, approximate=True)
print(result)