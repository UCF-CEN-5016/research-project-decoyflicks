import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
seq_len = 16
embedding_dim = 64

q = torch.rand(batch_size, seq_len, embedding_dim).to(device)
k = torch.rand(batch_size, seq_len, embedding_dim).to(device)

rotary_embedding = RotaryEmbedding(dim=embedding_dim, learned_freq=True, freqs_for='lang').to(device)
loss_fn = nn.MSELoss()

rotated_q, rotated_k = rotary_embedding.rotate_queries_or_keys(q, k)
dummy_loss = loss_fn(rotated_q, rotated_k)

dummy_loss.backward()
optimizer = torch.optim.Adam(rotary_embedding.parameters())
optimizer.step()
optimizer.zero_grad()

try:
    dummy_loss.backward()
except RuntimeError as e:
    assert 'Trying to backward through the graph a second time' in str(e)