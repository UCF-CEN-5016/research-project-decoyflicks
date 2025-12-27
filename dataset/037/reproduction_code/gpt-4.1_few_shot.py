import torch
import math

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, v, cos, sin):
    # Intended single rotation for q and v
    q_rot = (q * cos) + (rotate_half(q) * sin)
    v_rot_once = (v * cos) + (rotate_half(v) * sin)
    
    # Incorrect double rotation for v (simulating the bug)
    v_rot_twice = (v_rot_once * cos) + (rotate_half(v_rot_once) * sin)
    
    return q_rot, v_rot_once, v_rot_twice

# Setup sample tensors
batch_size, seq_len, dim = 2, 4, 8
q = torch.randn(batch_size, seq_len, dim)
v = torch.randn(batch_size, seq_len, dim)

# Create rotary embedding cos and sin (broadcastable to q,v)
theta = 10000 ** (torch.arange(0, dim, 2).float() / dim)
pos = torch.arange(seq_len).unsqueeze(1)
freqs = 1.0 / theta
angles = pos * freqs.T
cos = torch.cos(angles).repeat_interleave(2, dim=-1).unsqueeze(0)
sin = torch.sin(angles).repeat_interleave(2, dim=-1).unsqueeze(0)

q_rot, v_rot_once, v_rot_twice = apply_rotary_pos_emb(q, v, cos, sin)

print("Original v[0,0]:", v[0,0])
print("Value after one rotation[0,0]:", v_rot_once[0,0])
print("Value after two rotations (bug) [0,0]:", v_rot_twice[0,0])