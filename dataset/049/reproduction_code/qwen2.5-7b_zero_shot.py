import torch

shape = (100, 100)
bias = torch.zeros(shape)
print(torch.sum(bias))