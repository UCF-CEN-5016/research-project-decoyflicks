import torch

def cdist(x, y):
    x_norm = (x**2).sum(-1).view(-1, 1)
    y_norm = (y**2).sum(-1).view(1, -1)
    dist = x_norm + y_norm - 2 * torch.matmul(x, y.T)
    return dist

x = torch.randn(1000, 128)
y = torch.randn(1000, 128)

dist = cdist(x, y)
print(dist)

dist_clamped = cdist(x, y)
dist_clamped = torch.clamp(dist_clamped, min=0)
print(dist_clamped)