import torch

def compute_attention(query, key, value):
    # Transpose key to correct its shape
    key = key.transpose(1, 2)
    
    # Compute attention
    attention = torch.matmul(query, key) / torch.sqrt(torch.tensor(3.))
    
    return attention

# Simulated query, key, and value tensors with shape (batch, seq_len, embed_dim)
query = torch.randn(1, 2, 3)  # (batch=1, seq_len=2, embed_dim=3)
key = torch.randn(1, 3, 2)    # Corrected key shape by transposing
value = torch.randn(1, 2, 3)  # (batch=1, seq_len=2, embed_dim=3)

# Cross-attention computation with corrected shapes
attention = compute_attention(query, key, value)
print("Attention shape:", attention.shape)