import torch

def rotate_queries_and_keys(Q, K, pos):
    # Assume Q and K are of shape (seq_len, head_dim)
    seq_len, head_dim = Q.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    # Split into even and odd parts
    Q_even = Q[:, 0::2]
    Q_odd = Q[:, 1::2]
    K_even = K[:, 0::2]
    K_odd = K[:, 1::2]

    # Compute rotation angles (simplified for illustration)
    # This is not a standard RoPE implementation, but just for demonstration
    theta = pos / 10000  # This is a simplified theta calculation

    # Apply rotation: Q_rot = Q_even * cos(theta) - Q_odd * sin(theta)
    # For K, we'll simulate a faulty operation that divides by pos
    K_rot = (K_even * torch.cos(theta) - K_odd * torch.sin(theta)) / pos

    # Reconstruct the rotated tensor
    K_rot = torch.cat([
        K_rot[:, 0::2],  # Even indices
        K_rot[:, 1::2]   # Odd indices
    ], dim=1)

    return Q, K_rot

# Test case that causes NaNs
Q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
K = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
pos = torch.tensor([0.0])  # Zero value causes division by zero

# Apply the function
Q_rot, K_rot = rotate_queries_and_keys(Q, K, pos)

print("K_rot after rotation:", K_rot)

K_rot = (K_even * torch.cos(theta) - K_odd * torch.sin(theta)) / pos

# Add a small epsilon to prevent division by zero
eps = 1e-10
K_rot = (K_even * torch.cos(theta) - K_odd * torch.sin(theta)) / (pos + eps)