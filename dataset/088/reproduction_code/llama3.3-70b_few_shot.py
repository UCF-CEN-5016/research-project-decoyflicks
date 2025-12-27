import torch
import torch.nn.functional as F

def align_right(x, max_len, pad_id=0):
    """Aligns input tensor to the right by padding with zeros."""
    # Incorrectly hardcoded pad value
    return F.pad(x, (0, max_len - x.shape[1]), value=0)

# Sample data
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Align with incorrect pad value (0)
aligned_x = align_right(x, max_len=5, pad_id=10)
print("Aligned tensor with incorrect pad value (0):\n", aligned_x)

# Expected align function with correct pad value
def align_right_correct(x, max_len, pad_id):
    """Aligns input tensor to the right by padding with specified value."""
    return F.pad(x, (0, max_len - x.shape[1]), value=pad_id)

# Align with correct pad value (10)
aligned_x_correct = align_right_correct(x, max_len=5, pad_id=10)
print("Aligned tensor with correct pad value (10):\n", aligned_x_correct)