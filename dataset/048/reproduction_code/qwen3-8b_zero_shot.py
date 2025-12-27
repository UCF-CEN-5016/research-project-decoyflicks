import torch

t1 = torch.randn(1, 1649)
t2 = torch.randn(1, 1799)
emissions_arr = [t1, t2]
emissions = torch.cat(emissions_arr, dim=1).squeeze()