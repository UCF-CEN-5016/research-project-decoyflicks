import torch

def apply_rotation(tensor, theta, num_rotations):
    """
    Simulates applying rotation operation multiple times to a tensor.
    """
    for _ in range(num_rotations):
        tensor = tensor * torch.cos(theta)
    return tensor

# Create a sample tensor (shape: (seq_len, d_model))
tensor = torch.randn(10, 64)
theta = 0.1  # Rotation angle

# Apply rotation once and twice
rotated_once = apply_rotation(tensor, theta, 1)
rotated_twice = apply_rotation(tensor, theta, 2)

# Output the results
print("Original Tensor:\n", tensor)
print("After one rotation:\n", rotated_once)
print("After two rotations:\n", rotated_twice)
print("Difference between one and two rotations:\n", rotated_twice - rotated_once)