import torch

# Create a dummy model and tensors for reproduction
batch_size = 2
seq_length = 5
d_model = 32
num_patches = 4

# Pos embeddings shape: [max_positions, d_model]
pos_embedding = torch.randn(10, d_model)  # Example with max_positions=10

# Tokens tensor: [batch_size, seq_length, d_model]
tokens = torch.randn(batch_size, seq_length, d_model)

# Reproduce the bug line
tokens = tokens + pos_embedding[:, 1:(num_patches + 1)]

print("After modification:")
print(tokens)