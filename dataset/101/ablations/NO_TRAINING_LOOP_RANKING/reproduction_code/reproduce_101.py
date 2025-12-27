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

if torch.isnan(rotated_k).any():
    print('NaNs detected in rotated_k')

assert torch.isnan(rotated_k).any() == True

# Assuming a transformer model is defined elsewhere
# output = transformer_model(rotated_q, rotated_k)
# loss = loss_function(output)

# Check for NaN loss
# assert torch.isnan(loss) == True
# print('NaN loss detected')