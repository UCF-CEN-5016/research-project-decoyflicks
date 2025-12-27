import torch
import torch.nn as nn
import torch.optim as optim
from rotary_embedding_torch.rotary_embedding_torch import RotaryEmbedding

torch.manual_seed(42)

dim = 64
seq_len = 16
batch_size = 2

rotary_embedding = RotaryEmbedding(
    dim=dim,
    learned_freq=False,
    freqs_for='lang',
    use_xpos=False,
)

linear = nn.Linear(dim, dim)
optimizer = optim.Adam(list(linear.parameters()) + list(rotary_embedding.parameters()), lr=1e-3)

x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

for i in range(2):
    optimizer.zero_grad()
    q = linear(x)
    k = linear(x)
    rotated_q, rotated_k = rotary_embedding.rotate_queries_with_cached_keys(q, k)
    loss = rotated_q.mean()
    loss.backward()
    optimizer.step()