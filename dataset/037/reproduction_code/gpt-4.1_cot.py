import torch

def rotate_half(x):
    # rotate half dims: split last dim half, negate second half and swap halves
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, sin, cos):
    # x: [..., dim]
    # sin, cos: [..., dim]
    # Apply rotation as per RoPE
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return x_rotated

# Dummy tensor for value embedding
batch_size = 2
seq_len = 4
dim = 8  # must be even for rotary pos emb

value_emb = torch.randn(batch_size, seq_len, dim)

# Dummy sin and cos tensors mimicking positional embeddings
# For simplicity, let's create ones and zeros
sin = torch.zeros(seq_len, dim)
cos = torch.ones(seq_len, dim)

# Expand sin, cos to batch size for broadcasting
sin = sin.unsqueeze(0).expand(batch_size, -1, -1)
cos = cos.unsqueeze(0).expand(batch_size, -1, -1)

# Apply rotation once
value_once = apply_rotary_pos_emb(value_emb, sin, cos)

# Apply rotation twice
value_twice = apply_rotary_pos_emb(value_once, sin, cos)

print("Original value embedding:\n", value_emb)
print("\nValue embedding after one rotation:\n", value_once)
print("\nValue embedding after two rotations:\n", value_twice)

# Check if double rotation is the same as rotating once (should not be)
print("\nDifference between one rotation and two rotations:\n", (value_twice - value_once).abs().max())