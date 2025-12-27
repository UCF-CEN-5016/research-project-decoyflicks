import torch
from torch import einsum
from torch.cdist import cdist

def buggy_cdist(x, y):
    # This reproduces the problematic computation
    x2 = einsum('... d, ... d -> ...', x, x)
    y2 = einsum('... d, ... d -> ...', y, y)
    xy = einsum('... d, ... d -> ...', x, y)
    return (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2 * xy).sqrt()

def safe_cdist(x, y):
    # This shows the fixed version with clamping
    x2 = einsum('... d, ... d -> ...', x, x)
    y2 = einsum('... d, ... d -> ...', y, y)
    xy = einsum('... d, ... d -> ...', x, y)
    return (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2 * xy).clamp(min=0).sqrt()

# Create large input values that trigger the bug
x = torch.randn(3, 512) * 1e6
y = torch.randn(5, 512) * 1e6

# Demonstrate the bug
buggy_result = buggy_cdist(x, y)
print("Buggy result contains nans:", torch.isnan(buggy_result).any())

# Show fixed version
safe_result = safe_cdist(x, y)
print("Safe result contains nans:", torch.isnan(safe_result).any())

# Compare with torch's native cdist for reference
torch_result = cdist(x, y)
print("Torch cdist contains nans:", torch.isnan(torch_result).any())