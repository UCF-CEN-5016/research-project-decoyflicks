import torch

# Minimal setup to reproduce the bug
tokens = torch.randn(2, 16, 64)  # [batch_size, num_tokens, embed_dim]
num_patches = 16
pos_embedding = torch.randn(1, 17, 64)  # [1, num_positions, embed_dim]

# Problematic line from the original code
tokens = tokens + pos_embedding[:, 1:(num_patches + 1)]

# This will raise a shape mismatch error:
# RuntimeError: shape [2,16,64] does not match [1,16,64]

tokens = tokens + pos_embedding[:, 1:(num_patches + 1)].expand(tokens.shape[0], -1, -1)