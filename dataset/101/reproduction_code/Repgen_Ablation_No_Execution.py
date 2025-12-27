import torch
from rotary_embedding_torch import RotaryEmbedding

seq_length = 1024
batch_size = 32
dim = 512

rotary_emb = RotaryEmbedding(dim=dim, use_xpos=True, xpos_scale_base=512)

Q = torch.rand((batch_size, seq_length, dim))
K = torch.rand((batch_size, seq_length, dim))

Q_rotated, K_rotated = rotary_emb.rotate_queries_and_keys(Q, K)

assert not torch.isnan(K_rotated).any().item(), "K tensor contains NaN values"