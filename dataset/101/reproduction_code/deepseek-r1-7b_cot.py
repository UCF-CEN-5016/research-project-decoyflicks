import torch
from torch.nn import functional as F

# Set device and parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
seq_len = 1024
dim = 512
num_heads = 8

# Create example input tensors Q, K with small positive values to trigger the issue
Q = torch.randn(batch_size, seq_len, dim, device=device) * 1e-6
K = torch.randn(batch_size, seq_len, dim, device=device) * 1e-6

# Apply positional encoding (xpos)
angle = torch.arange(0., seq_len, dtype=torch.float32, device=device).unsqueeze(-1) * 10.0 / seq_len
xpos = F.sin(angle)

# Function to rotate queries and keys using the provided xpos
def rotate_queries_and_keys(q, k, angle):
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    
    # Perform rotation for each query and key vector in K (element-wise across batches)
    rotated_q = q * cos_theta - k * sin_theta
    rotated_k = q * sin_theta + k * cos_theta
    
    return rotated_q, rotated_k

# Apply the rotation
Q_rotated, K_rotated = rotate_queries_and_keys(Q, K, angle)

# Check for NaNs in K after rotation
print("NaNs in K after rotation:", torch.isnan(K_rotated).any())

# Example fix: Add a small epsilon to avoid division by zero (if used later)
epsilon = 1e-8
K_rotated_safe = K_rotated + epsilon

# After fixing, if needed:
# Ensure all layers are safe from NaNs and handle invalid operations