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

# Instantiate AutoregressiveWrapper with a dummy net
class DummyNet(torch.nn.Module):
    def forward(self, x):
        return x

net = DummyNet()
wrapper = AutoregressiveWrapper(net, pad_value=5)

# Call generate method
output = wrapper.generate(prompts=t, seq_len=10, prompt_lens=lens)

# Inside align_right function
def align_right(t, lens, pad_id=0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype
    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len
    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()
    batch_arange = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device=device, dtype=torch.long)
    
    # Use pad_id instead of hardcoded value 0 for padding
    t = F.pad(t, (max_pad_len, 0), value=pad_id)  # Here, pad_id is now used
    offset = max_pad_len - pad_lens
    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# Capture output of align_right
aligned_output = align_right(t, lens, pad_id=wrapper.pad_value)

# Print output tensor and expected tensor
print("Output Tensor:", aligned_output)
print("Expected Tensor (with pad_value):", F.pad(t, (max_pad_len, 0), value=5))