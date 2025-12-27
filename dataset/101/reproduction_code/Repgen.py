import torch
from rotary_embedding_torch import RotaryEmbedding

batch_size = 32
seq_len = 512
dim = 768

Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)

rotary_emb = RotaryEmbedding(dim, use_xpos=True, xpos_scale_base=512)
Q_rotated, K_rotated = rotary_emb.rotate_queries_and_keys(Q, K)

# Verify that after applying rotary embeddings to Q and K, K contains a very small fraction of NaNs
print(torch.isnan(K_rotated).sum().item() / (batch_size * seq_len * dim))