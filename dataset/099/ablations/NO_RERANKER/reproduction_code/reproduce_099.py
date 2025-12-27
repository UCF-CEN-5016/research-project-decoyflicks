import torch
import torch.nn as nn
from rotary_embedding_torch.rotary_embedding_torch import RotaryEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
seq_length = 128
embedding_dim = 64

rotary_embedding = RotaryEmbedding(dim=64, learned_freq=True, use_xpos=True).to(device)
queries = torch.randn(batch_size, seq_length, embedding_dim, device=device)
keys = torch.randn(batch_size, seq_length, embedding_dim, device=device)

with torch.cuda.amp.autocast(enabled=True):
    rotated_queries, rotated_keys = rotary_embedding.rotate_queries_with_cached_keys(queries, keys)
    target = torch.randn_like(rotated_queries)
    loss = nn.MSELoss()(rotated_queries, target)

assert torch.isnan(loss).any()
print("Loss value:", loss.item())