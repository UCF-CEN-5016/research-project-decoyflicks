import torch
import torch.nn.functional as F

def align_right(tensor, pad_id):
    # The pad_id is not used, and value=0 is hardcoded
    return F.pad(tensor, (0, 1, 0, 0), value=0)

# Minimal test case
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
pad_id = 999  # Expected to be used for padding, but ignored

# Apply the buggy align_right function
padded_tensor = align_right(tensor, pad_id)

print("Padded Tensor:\n", padded_tensor)
print("Used padding value:", F.pad(tensor, (0, 1, 0, 0), value=0).tolist()[-1])