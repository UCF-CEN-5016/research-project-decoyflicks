import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

batch_size = 8
seq_len = 16
embedding_dim = 64

input_tensor = torch.rand(batch_size, seq_len, embedding_dim)
rotary_embedding = RotaryEmbedding(dim=embedding_dim, learned_freq=True, freqs_for='lang')
loss_fn = nn.MSELoss()
target_tensor = torch.rand(batch_size, seq_len, embedding_dim)

output = rotary_embedding(input_tensor, seq_len=seq_len)
loss = loss_fn(output, target_tensor)
loss.backward()

optimizer = torch.optim.Adam(rotary_embedding.parameters())
optimizer.step()

output = rotary_embedding(input_tensor, seq_len=seq_len)
loss = loss_fn(output, target_tensor)
loss.backward()  # This should raise RuntimeError