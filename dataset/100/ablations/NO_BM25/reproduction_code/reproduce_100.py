import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

batch_size = 8
seq_len = 16

rotary_embedding = RotaryEmbedding(dim=32, learned_freq=True, freqs_for='lang')
queries = torch.randn(batch_size, seq_len, 32)
keys = torch.randn(batch_size, seq_len, 32)

rotated_queries, rotated_keys = rotary_embedding.rotate_queries_and_keys(queries, keys)
loss = rotated_queries.sum()

loss.backward()

rotated_queries, rotated_keys = rotary_embedding.rotate_queries_and_keys(queries, keys)

try:
    loss.backward()
except RuntimeError as e:
    assert 'Trying to backward through the graph a second time' in str(e)