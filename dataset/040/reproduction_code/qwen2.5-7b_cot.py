import torch

# Setup
batch_size = 1
seq_len = 4
embed_dim = 8

# Create sample queries and keys
Q = torch.randn(batch_size, seq_len, embed_dim)
K = torch.randn(batch_size, seq_len, embed_dim)

# Compute attention scores using matrix multiplication and transpose
attn_scores = torch.bmm(Q, K.transpose(1, 2))

# Apply softmax correctly along the last dimension
correct_attn = torch.nn.functional.softmax(attn_scores, dim=-1)

# Apply softmax incorrectly along the second dimension
incorrect_attn = torch.nn.functional.softmax(attn_scores, dim=1)

# Compare the results
print("Correct softmax (dim=2):")
print(correct_attn)

print("\nIncorrect softmax (dim=1):")
print(incorrect_attn)