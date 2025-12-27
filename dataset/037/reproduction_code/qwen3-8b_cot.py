import torch

def apply_rope_once(tensor, theta):
    """
    Simulates a simple rotation operation applied once to a tensor.
    """
    # Apply the rotation (simplified as multiplication by cos(theta))
    return tensor * torch.cos(theta)

def apply_rope_twice(tensor, theta):
    """
    Simulates applying the rotation twice.
    """
    # Apply rotation once
    rotated_once = apply_rope_once(tensor, theta)
    # Apply rotation again
    rotated_twice = apply_rope_once(rotated_once, theta)
    return rotated_twice

# Create a sample tensor (shape: (seq_len, d_model))
tensor = torch.randn(10, 64)
theta = 0.1  # Rotation angle

# Apply rotation once and twice
rotated_once = apply_rope_once(tensor, theta)
rotated_twice = apply_rope_twice(tensor, theta)

# Output the results
print("Original Tensor:\n", tensor)
print("After one rotation:\n", rotated_once)
print("After two rotations:\n", rotated_twice)
print("Difference between one and two rotations:\n", rotated_twice - rotated_once)