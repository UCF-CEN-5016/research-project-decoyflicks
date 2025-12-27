import torch

def split_into_even_and_odd_parts(tensor):
    even = tensor[:, 0::2]
    odd = tensor[:, 1::2]
    return even, odd

def apply_rotation(tensor_even, tensor_odd, theta):
    rotated = (tensor_even * torch.cos(theta) - tensor_odd * torch.sin(theta))
    return rotated

def rebuild_tensor(rotated_even, rotated_odd):
    rebuilt = torch.cat([rotated_even, rotated_odd], dim=1)
    return rebuilt

def rotate_queries_and_keys(Q, K, pos):
    seq_len, head_dim = Q.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    Q_even, Q_odd = split_into_even_and_odd_parts(Q)
    K_even, K_odd = split_into_even_and_odd_parts(K)

    theta = pos / 10000

    K_rotated = apply_rotation(K_even, K_odd, theta) / (pos + 1e-10)

    K_rot = rebuild_tensor(K_rotated[:, 0::2], K_rotated[:, 1::2])

    return Q, K_rot

# Test case that causes NaNs
Q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
K = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
pos = torch.tensor([0.0])  # Zero value causes division by zero

# Apply the function
Q_rot, K_rot = rotate_queries_and_keys(Q, K, pos)

print("K_rot after rotation:", K_rot)