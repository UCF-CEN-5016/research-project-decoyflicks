import torch
import torch.nn.functional as F

def align_right(x, pad_id, max_len):
    return F.pad(x, (0, max_len - x.size(1)), value=0)

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
pad_id = 10
max_len = 5

result = align_right(x, pad_id, max_len)
print(result)