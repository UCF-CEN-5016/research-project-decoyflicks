import torch
import torch.nn.functional as F

def align_right(tensor, pad_id, max_len):
    pad_size = max_len - tensor.size(1)
    padded_tensor = F.pad(tensor, (0, pad_size), value=pad_id)
    return padded_tensor

tensor = torch.tensor([[1, 2, 3]])
max_len = 5
pad_id = 1
result = align_right(tensor, pad_id, max_len)
print(result)