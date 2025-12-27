import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

torch.manual_seed(42)

batch_size = 8
embedding_dim = 64
seq_length = 10

Q = torch.rand(batch_size, seq_length, embedding_dim)
K = torch.rand(batch_size, seq_length, embedding_dim)

rotary_emb = RotaryEmbedding(dim=embedding_dim, use_xpos=True)

rotated_Q, rotated_K = rotary_emb.rotate_queries_and_keys(Q, K)

assert torch.isnan(rotated_K).any() == True
print(rotated_K)

dummy_output = torch.rand(batch_size, seq_length, embedding_dim)
dummy_target = torch.rand(batch_size, seq_length, embedding_dim)

loss = nn.MSELoss()(dummy_output, dummy_target)

assert torch.isnan(loss).any() == True
print(loss)