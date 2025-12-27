import torch
import torch.nn.functional as F
from x_transformers.autoregressive_wrapper import align_right

batch_size = 4
seq_len = 10
embedding_dim = 16

t = torch.randn(batch_size, seq_len, embedding_dim)
lens = torch.randint(1, seq_len + 1, (batch_size,))
pad_id = 0

aligned_output = align_right(t, lens, pad_id)

assert lens.ndim == 1 and lens.shape[0] == batch_size
pad_lens = seq_len - lens
assert pad_lens.shape == (batch_size,)
max_pad_len = pad_lens.amax()
assert isinstance(max_pad_len, int)

batch_arange = torch.arange(batch_size, device=t.device, dtype=torch.long)[..., None]
prompt_len_arange = torch.arange(seq_len, device=t.device, dtype=torch.long)

padded_t = F.pad(t, (max_pad_len, 0), value=0)
assert pad_id not in padded_t

print(aligned_output)