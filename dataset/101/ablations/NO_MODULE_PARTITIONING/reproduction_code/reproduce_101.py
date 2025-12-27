import torch
from rotary_embedding_torch import RotaryEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dim = 64
rotary_emb = RotaryEmbedding(dim=dim, use_xpos=True, xpos_scale_base=512)

batch_size = 2
seq_len = 10
Q = torch.randn(batch_size, seq_len, dim, device=device)
K = torch.randn(batch_size, seq_len, dim, device=device)

rotated_q, rotated_k = rotary_emb.rotate_queries_and_keys(Q, K)

if torch.isnan(K).any():
    print("NaNs detected in K after rotation.")

loss_fn = torch.nn.MSELoss()
dummy_target = torch.randn(batch_size, seq_len, dim, device=device)
loss = loss_fn(rotated_q, dummy_target)

assert torch.isnan(loss).item(), "Loss is not NaN."