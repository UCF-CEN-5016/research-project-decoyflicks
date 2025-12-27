import torch
import torch.nn as nn

class SimpleSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        q = self.query(x)  # (batch, seq_len, dim)
        k = self.key(x)    # (batch, seq_len, dim)
        v = self.value(x)  # (batch, seq_len, dim)

        # Compute attention scores: (batch, seq_len, seq_len)
        attn = torch.einsum('bqd,bkd->bqk', q, k)

        # Incorrect softmax dimension (dim=1 instead of dim=2)
        attn_incorrect = attn.softmax(dim=1)  # Should be dim=2

        out_incorrect = torch.einsum('bqk,bkd->bqd', attn_incorrect, v)

        # Correct softmax dimension
        attn_correct = attn.softmax(dim=2)
        out_correct = torch.einsum('bqk,bkd->bqd', attn_correct, v)

        return out_incorrect, out_correct

# Sample input
x = torch.randn(2, 4, 8)  # batch=2, seq_len=4, dim=8

attention = SimpleSelfAttention(dim=8)
out_incorrect, out_correct = attention(x)

print("Output with incorrect softmax dim:", out_incorrect)
print("Output with correct softmax dim:", out_correct)