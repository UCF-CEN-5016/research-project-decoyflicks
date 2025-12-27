import torch
import torch.nn.functional as F

def align_right(x, pad_id=1):
    # Original buggy implementation
    lengths = (x != 0).long().sum(dim=1)
    max_len = x.shape[1]
    pad_len = max_len - lengths
    pad_left = torch.zeros_like(x)
    pad_left[torch.arange(x.size(0)), pad_len] = 1  # Bug: Hardcoded pad value 1
    x_padded = F.pad(x, (0, 0), value=0)  # Bug: Hardcoded pad value 0
    return x_padded

# Test case
x = torch.tensor([
    [2, 3, 0, 0],  # Sequence 1
    [4, 5, 6, 0]    # Sequence 2
])
pad_id = 9  # Should be used but isn't

# Expected: Right-aligned with pad_id=9
# Actual: Right-aligned with pad_id=0
result = align_right(x, pad_id)
print("Result (buggy):")
print(result)