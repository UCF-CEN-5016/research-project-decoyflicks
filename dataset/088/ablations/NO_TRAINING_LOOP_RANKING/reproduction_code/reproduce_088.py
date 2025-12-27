import torch
import torch.nn.functional as F
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# Setup
batch_size = 4
seq_len = 10
embedding_dim = 16

# Create random tensor
t = torch.rand(batch_size, seq_len, embedding_dim)

# Create lens tensor
lens = torch.randint(1, seq_len + 1, (batch_size,))

# Instantiate AutoregressiveWrapper
dummy_net = torch.nn.Linear(embedding_dim, embedding_dim)  # Dummy net
wrapper = AutoregressiveWrapper(net=dummy_net, pad_value=1)

# Call generate method
output = wrapper.generate(prompts=t, seq_len=5, prompt_lens=lens)

# Check align_right function
def align_right(t, lens, pad_id=0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device=device, dtype=torch.long)

    t = F.pad(t, (max_pad_len, 0), value=0)  # Incorrect usage of pad_id
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# Verify output
aligned_output = align_right(t, lens, pad_id=wrapper.pad_value)
print(aligned_output)