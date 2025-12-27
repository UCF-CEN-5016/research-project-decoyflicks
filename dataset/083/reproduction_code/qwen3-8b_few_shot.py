import torch

# Simulated query, key, and value tensors with shape (batch, seq_len, embed_dim)
query = torch.randn(1, 2, 3)  # (batch=1, seq_len=2, embed_dim=3)
key = torch.randn(1, 2, 3)    # Incorrectly shaped key (not transposed)
value = torch.randn(1, 2, 3)  # (batch=1, seq_len=2, embed_dim=3)

# Cross-attention computation with shape mismatch
# Mistake: key is not transposed, leading to dimension mismatch
attention = query @ key / torch.sqrt(torch.tensor(3))  # This will raise an error
print("Attention shape:", attention.shape)