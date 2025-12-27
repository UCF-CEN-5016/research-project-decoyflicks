import torch

# 🔧 Setup
batch_size = 1
seq_len = 4  # Sequence length (e.g., height or width)
embed_dim = 8  # Embedding dimension

# 🧠 Create sample queries and keys
Q = torch.randn(batch_size, seq_len, embed_dim)
K = torch.randn(batch_size, seq_len, embed_dim)

# 🧮 Compute attention scores using einsum
attn_scores = torch.einsum('b i d, b j d -> b i j', Q, K)

# 🧠 Apply softmax on the correct dimension (dim=2)
correct_attn = attn_scores.softmax(dim=2)

# 🧠 Apply softmax on the incorrect dimension (dim=1)
incorrect_attn = attn_scores.softmax(dim=1)

# 📊 Compare the results
print("Correct softmax (dim=2):")
print(correct_attn)

print("\nIncorrect softmax (dim=1):")
print(incorrect_attn)