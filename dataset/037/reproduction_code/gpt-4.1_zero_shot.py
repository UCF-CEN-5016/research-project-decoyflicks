import torch

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, v, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    v_rot = (v * cos) + (rotate_half(v) * sin)
    v_rot_again = (v_rot * cos) + (rotate_half(v_rot) * sin)
    return q_rot, v_rot, v_rot_again

batch, seq_len, dim = 1, 2, 4
q = torch.arange(batch * seq_len * dim, dtype=torch.float32).reshape(batch, seq_len, dim)
v = torch.arange(batch * seq_len * dim, dtype=torch.float32).reshape(batch, seq_len, dim) + 100
cos = torch.ones(batch, seq_len, dim)
sin = torch.ones(batch, seq_len, dim) * 0.5

q_rot, v_rot, v_rot_again = apply_rotary_pos_emb(q, v, cos, sin)

print("v original:\n", v)
print("v rotated once:\n", v_rot)
print("v rotated twice:\n", v_rot_again)