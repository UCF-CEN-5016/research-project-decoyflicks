import torch

t1 = torch.randn(1, 1649)
t2 = torch.randn(1, 1799)
emissions = torch.cat((t1, t2), dim=1).squeeze()