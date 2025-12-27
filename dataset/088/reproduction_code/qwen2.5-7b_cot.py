import torch
import torch.nn.functional as F

def align_right(tensor):
    return F.pad(tensor, (0, 1, 0, 0))

# Minimal test case
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Apply the align_right function
padded_tensor = align_right(tensor)

print("Padded Tensor:\n", padded_tensor)
print("Used padding value:", padded_tensor[:, -1].tolist())