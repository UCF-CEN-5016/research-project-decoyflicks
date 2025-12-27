import torch
import torch.nn.functional as F

def align_right(x, lens, pad_id=1):
    batch, seq_len = x.shape
    arange = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
    padded = pad_id * torch.ones_like(x)
    padded = padded.scatter(1, (seq_len - lens).unsqueeze(1) + arange, x)
    mask = arange >= (seq_len - lens).unsqueeze(1)
    padded = padded.masked_fill(~mask, pad_id)
    return F.pad(padded, (0, 1), value=0)

x = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
lens = torch.tensor([3, 2])
result = align_right(x, lens, pad_id=1)
print(result)