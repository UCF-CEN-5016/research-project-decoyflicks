import torch
import torch.nn.functional as F

def rotary_emb_rotate_queries_and_keys(Q, K, sin, cos):
    # Simplified rotary embedding rotation logic
    # Q, K: [batch, seq_len, num_heads, head_dim]
    # sin, cos: [seq_len, 1, head_dim]
    
    # Split Q and K into even and odd parts for rotation
    Q1, Q2 = Q[..., ::2], Q[..., 1::2]
    K1, K2 = K[..., ::2], K[..., 1::2]
    
    # Rotate Q
    Q_rotated = torch.empty_like(Q)
    Q_rotated[..., ::2] = Q1 * cos - Q2 * sin
    Q_rotated[..., 1::2] = Q1 * sin + Q2 * cos
    
    # Rotate K
    K_rotated = torch.empty_like(K)
    K_rotated[..., ::2] = K1 * cos - K2 * sin
    K_rotated[..., 1::2] = K1 * sin + K2 * cos
    
    return Q_rotated, K_rotated

# Setup
batch_size, seq_len, num_heads, head_dim = 2, 4, 2, 8

# Queries and Keys with random floats
Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
K = torch.randn(batch_size, seq_len, num_heads, head_dim)

# Create sin and cos with shape [seq_len, 1, head_dim]
# Using small values to simulate edge case causing NaNs
angles = torch.linspace(0, 1e10, seq_len).unsqueeze(1).unsqueeze(2)
sin = torch.sin(angles)
cos = torch.cos(angles)

# Apply rotary embeddings rotation
Q_rot, K_rot = rotary_emb_rotate_queries_and_keys(Q, K, sin, cos)

# Check for NaNs in K_rot
print("NaNs in rotated K:", torch.isnan(K_rot).sum().item())

# Downstream loss example
output = Q_rot @ K_rot.transpose(-2, -1)  # [batch, seq_len, num_heads, seq_len]
loss = output.mean()
print("Loss:", loss.item())