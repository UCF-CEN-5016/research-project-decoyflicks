import torch

def cdist(x, y):
    # Proper implementation with clamping to prevent negative values
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)
    xy = x @ y.transpose(-2, -1)
    distances = torch.clamp(x_sq + y_sq.transpose(-2, -1) - 2 * xy, min=0)
    return distances

# Create inputs
x = torch.randn(100, 512) * 1e6  # Large values increase overflow risk
y = torch.randn(100, 512) * 1e6

# Compute distances
distances = cdist(x, y)
print("Negative count after clamping:", (distances < 0).sum().item())