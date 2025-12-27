import torch

def cdist(x, y):
    x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.matmul(x, y.transpose(-2, -1))
    
    distances = torch.clamp(x_sq + y_sq.transpose(-2, -1) - 2 * xy, min=0)
    return distances

# Create inputs
x = torch.randn(100, 512) * 1e6
y = torch.randn(100, 512) * 1e6

# Compute distances with proper implementation
distances = cdist(x, y)
print("NaN count:", torch.isnan(distances).sum().item())