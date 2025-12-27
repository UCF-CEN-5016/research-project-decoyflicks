import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a minimal self-attention module
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Compute query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute attention scores
        attn = torch.einsum('bij,bkj->bik', q, k)

        # Incorrect axis for softmax (should be dim=2)
        attn = attn.softmax(dim=1)  # <--- BUG INTRODUCED HERE

        # Compute output
        output = torch.einsum('bik,bkj->bij', attn, v)
        return output

# Create a sample input
embed_dim = 10
batch_size = 32
seq_len = 20
x = torch.randn(batch_size, seq_len, embed_dim)

# Create a self-attention module
attn_module = SelfAttention(embed_dim)

# Run the self-attention module
output = attn_module(x)

# Print the output shape
print(f"Output shape: {output.shape}")