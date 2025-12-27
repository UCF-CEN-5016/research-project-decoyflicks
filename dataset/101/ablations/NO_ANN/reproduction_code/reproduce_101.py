import torch
from rotary_embedding_torch import RotaryEmbedding

torch.manual_seed(42)
dim = 64
rotary_emb = RotaryEmbedding(dim=dim, use_xpos=True)

batch_size = 8
seq_len = 16
Q = torch.rand(batch_size, seq_len, dim)
K = torch.rand(batch_size, seq_len, dim)

rotated_Q, rotated_K = rotary_emb.rotate_queries_and_keys(Q, K)

if torch.isnan(rotated_K).any():
    print("NaNs found in K after rotation.")

dummy_loss = ((rotated_Q - rotated_K) ** 2).mean()

if torch.isnan(dummy_loss).item():
    print("NaN loss detected.")

print(f"Shapes - Q: {Q.shape}, K: {K.shape}, Loss: {dummy_loss.item()}")