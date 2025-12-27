import torch

def rotate_queries_and_keys(Q, K, pos):
    return Q / pos, K / pos

# Create example tensors
Q = torch.randn(1, 10, 64)  # Query tensor
K = torch.randn(1, 10, 64)  # Key tensor
pos = torch.tensor([0, 1, 2, 3, 4])  # Position tensor with some zero values

# Apply the rotation function
Q_rot, K_rot = rotate_queries_and_keys(Q, K, pos)

# Check how many NaNs are in K_rot
nan_count = torch.isnan(K_rot).sum()
print(f"Number of NaNs in K_rot: {nan_count}")