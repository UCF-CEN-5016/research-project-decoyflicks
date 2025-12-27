import torch
from rotary_embedding_torch import RotaryEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dim = 64
rotary_emb = RotaryEmbedding(dim=dim, use_xpos=True)

batch_size = 2
seq_len = 10
Q = torch.randn(batch_size, seq_len, dim, device=device)
K = torch.randn(batch_size, seq_len, dim, device=device)

rotated_Q, rotated_K = rotary_emb.rotate_queries_and_keys(Q, K)

print(f"Rotated Q shape: {rotated_Q.shape}, Rotated K shape: {rotated_K.shape}")

if torch.isnan(rotated_K).any():
    print("NaNs detected in K")

dummy_loss = torch.mean((rotated_Q - rotated_K) ** 2)

if torch.isnan(dummy_loss).item():
    print("NaN loss detected")

print(f"Shapes - Q: {Q.shape}, K: {K.shape}, Loss: {dummy_loss.item()}")