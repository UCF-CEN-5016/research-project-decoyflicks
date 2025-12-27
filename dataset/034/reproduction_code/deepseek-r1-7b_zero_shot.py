import torch.nn.functional as F

x = torch.randn(3, 4)
F.gelu(x, approximate='True')