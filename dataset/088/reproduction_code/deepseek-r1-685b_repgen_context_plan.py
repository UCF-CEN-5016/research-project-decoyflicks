import torch
import torch.nn.functional as F

def align_right(x, pad_id=1):
    lengths = (x != 0).sum(dim=1)
    max_len = x.shape[1]
    pad_len = max_len - lengths
    pad_values = torch.full_like(x, pad_id)
    for i in range(x.size(0)):
        x[i, :pad_len[i]] = pad_values[i, :pad_len[i]]
    return x

# Test case
x = torch.tensor([
    [2, 3, 0, 0],  # Sequence 1
    [4, 5, 6, 0]    # Sequence 2
])
pad_id = 9

# Expected: Right-aligned with pad_id=9
result = align_right(x.clone(), pad_id)
print("Result (corrected):")
print(result)