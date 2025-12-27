import torch
from torch.nn.functional import rotate_queries_and_keys  # Assuming this is a custom function

# Example code to reproduce the bug
# Creates two random matrices Q and K for demonstration
batch_size = 32
seq_len = 100
d_k = 512  # Dimension of key vectors

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)

# Simulated rotation (this could be a simplified version of the actual function used)
def rotate_queries_and_keys(Q, K):
    """
    Simplified example of how such functions might operate.
    In practice, ensure proper handling to prevent NaNs.
    """
    # Example: Applying a random rotation with scaling factors
    scale = torch.randn(batch_size, 1) * 0.01  # Small random values can cause instability
    rotated_Q = Q * scale + (torch.randn_like(Q) * 1e-8)
    rotated_K = K * scale + (torch.randn_like(K) * 1e-8)
    return rotated_Q, rotated_K

# Applying the rotation and checking for NaNs in K after rotation
rotated_Q, rotated_K = rotate_queries_and_keys(Q, K)

if torch.isnan(rotated_K).any():
    print("NaN values detected in rotated_K after rotation operation.")
else:
    print("No NaN values present.")

# Example loss calculation (assuming this is part of the model's forward pass)
loss = rotated_K.mean()
print(f"Loss with NaNs: {loss.item()}")