import torch
import torch.nn.functional as F

def align_right(tensor, pad_id):
    """Function to pad a tensor with a specific pad_id value on the right side."""
    return F.pad(tensor, (0, 1), value=pad_id)

# Test case with a custom pad_id
tensor = torch.tensor([[1, 2], [3, 4]])
pad_id = 5
padded_tensor = align_right(tensor, pad_id)
print("Padded Tensor:", padded_tensor)
print("Padding value used:", padded_tensor[-1, -1])