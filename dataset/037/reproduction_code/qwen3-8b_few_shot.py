

import torch

def rotate(tensor):
    """Simulated rotation function (simplified for demonstration)"""
    dim = tensor.size(-1)
    half = dim // 2
    return torch.cat([tensor[..., half:], tensor[..., :half]], dim=-1)

# Value embedding tensor
value = torch.randn(1, 10, 16)  # shape: (batch, seq_len, dim)

# Apply rotation twice (as seen in the code)
rotated_twice = rotate(rotate(value))

# Apply rotation once (correct approach)
rotated_once = rotate(value)

print("After one rotation:")
print(rotated_once)
print("After two rotations:")
print(rotated_twice)