import torch
import torch.nn.functional as F

def align_right(tensor, pad_id):
    """Function that should use pad_id but does not."""
    return F.pad(tensor, (0, 1), value=0)

# Test case with a custom pad_id
tensor = torch.tensor([[1, 2], [3, 4]])
padded_tensor = align_right(tensor, pad_id=5)
print("Padded Tensor:", padded_tensor)
print("Padding value used:", padded_tensor[-1, -1])