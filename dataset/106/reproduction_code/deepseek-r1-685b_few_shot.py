import torch

def unsafe_cdist(x, y):
    # Problematic implementation without clamping
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)
    xy = x @ y.transpose(-2, -1)
    return x_sq + y_sq.transpose(-2, -1) - 2 * xy  # Can produce negative values

# Create inputs that trigger the issue
x = torch.randn(100, 512) * 1e6  # Large values increase overflow risk
y = torch.randn(100, 512) * 1e6

# Compute distances (will contain NaN)
distances = unsafe_cdist(x, y)
print("NaN count:", torch.isnan(distances).sum().item())

# Proper implementation with clamping would prevent this:
# distances = torch.clamp(x_sq + y_sq.transpose(-2, -1) - 2 * xy, min=0)