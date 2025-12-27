import torch
import torch.nn.functional as F

def align_right(x, pad_id):
    """
    Aligns the input tensor to the right by padding with the given pad_id.
    """
    # Simulate the buggy implementation
    padded_x = F.pad(x, (0, 10), value=0)  # Hardcoded value of 0 instead of pad_id
    return padded_x

# Create a sample tensor
x = torch.tensor([1, 2, 3])

# Set a pad_id value
pad_id = 5

# Call the align_right function with the pad_id
padded_x = align_right(x, pad_id)

# Print the result
print(padded_x)

# Expected output (if the bug is fixed):
# tensor([1, 2, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

# Actual output (with the bug):
# tensor([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])