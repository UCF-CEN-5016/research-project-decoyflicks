import torch
import torch.nn.functional as F

def align_right(x, pad_id):
    return F.pad(x, (1, 0), value=0)  # pad_id is not used here

x = torch.tensor([[1, 2, 3]])
pad_id = 9
out = align_right(x, pad_id)
print(out)