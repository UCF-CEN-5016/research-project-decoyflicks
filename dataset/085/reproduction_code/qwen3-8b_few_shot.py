import torch

# Create tokens and positional embeddings
tokens = torch.randn(2, 3, 64)  # batch_size=2, num_patches=3, d=64
pos_embedding = torch.randn(1, 4, 64)  # num_patches + 1 = 4

# Attempt to add positional embeddings to tokens
tokens = tokens + pos_embedding[:, 1:4]  # This will raise a shape mismatch error

print(tokens.shape)