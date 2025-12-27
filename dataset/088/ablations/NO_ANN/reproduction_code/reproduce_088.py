import torch
import torch.nn.functional as F
from x_transformers.autoregressive_wrapper import align_right

batch_size = 4
seq_len = 10
embedding_dim = 16

t = torch.rand(batch_size, seq_len, embedding_dim)
lens = torch.randint(1, seq_len + 1, (batch_size,))
pad_id = 5

aligned_output = align_right(t, lens, pad_id)

assert aligned_output.shape == (batch_size, seq_len, embedding_dim)
print("Output tensor:", aligned_output)