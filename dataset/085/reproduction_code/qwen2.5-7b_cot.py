import torch

# Minimal setup to reproduce the bug
tokens = torch.randn(2, 16, 64)  # [batch_size, num_tokens, embed_dim]
num_patches = 16
pos_embedding = torch.randn(1, 17, 64)  # [1, num_positions, embed_dim]

# Corrected line
pos_embedding_trimmed = pos_embedding[:, 1:(num_patches + 1)]
pos_embedding_expanded = pos_embedding_trimmed.expand(tokens.shape[0], -1, -1)
tokens = tokens + pos_embedding_expanded