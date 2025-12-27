import torch
from transformer import MAE

# Initialize MAE model with sample inputs
num_patches = 8
model = MAE(num_patches=num_patches)

# Create some tokens and positional embeddings
tokens = torch.randn(1, num_patches)
pos_embedding = torch.randn(1, num_patches + 1)

# Trigger the bug
model.tokens = tokens + pos_embedding[:, 1:(num_patches + 1)]

print("Model state:", model)