import torch

def safe_cdist(x, y):
    x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    xy = torch.matmul(x, y.transpose(-2, -1))
    distances = x_sq + y_sq.transpose(-2, -1) - 2 * xy
    distances = torch.clamp(distances, min=0)  # Using clamp to prevent negative values
    return distances

# Create inputs that trigger the issue
x = torch.randn(100, 512) * 1e6  # Large values increase overflow risk
y = torch.randn(100, 512) * 1e6

# Compute distances (will not contain NaN)
distances = safe_cdist(x, y)
print("Negative count:", (distances < 0).sum().item())