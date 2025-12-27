import torch
import torch.nn.functional as F

def align_right(t, lens, pad_id=0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device=device, dtype=torch.long)

    t = F.pad(t, (max_pad_len, 0), value=0)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

batch_size = 4
seq_len = 10
embedding_dim = 16
t = torch.randn(batch_size, seq_len, embedding_dim)
lens = torch.randint(1, seq_len + 1, (batch_size,))
pad_id = 0

aligned_output = align_right(t, lens, pad_id)
print(aligned_output.shape)
print(aligned_output)
assert (aligned_output == pad_id).sum() == 0