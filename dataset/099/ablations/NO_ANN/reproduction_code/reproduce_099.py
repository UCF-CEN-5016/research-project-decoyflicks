import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
seq_len = 128
embedding_dim = 64

q = torch.randn(batch_size, seq_len, embedding_dim, device=device)
k = torch.randn(batch_size, seq_len, embedding_dim, device=device)

model = RotaryEmbedding(dim=embedding_dim, use_xpos=True).to(device)
model.train()

target = torch.randint(0, 2, (batch_size,), device=device)
loss_fn = nn.CrossEntropyLoss()

rotated_q, rotated_k = model.rotate_queries_with_cached_keys(q, k)
loss = loss_fn(rotated_q.view(-1, embedding_dim), target)

if torch.isnan(loss).item():
    print("Loss is NaN:", loss.item())

torch.cuda.empty_cache()