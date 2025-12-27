import torch
import torch.nn.functional as F

def attention_bug_repro(q, k, v, buggy=True):
    """
    Reproduce the attention softmax dimension bug
    Args:
        q: query tensor [batch, heads, seq_len, dim]
        k: key tensor [batch, heads, seq_len, dim]
        v: value tensor [batch, heads, seq_len, dim]
        buggy: whether to use incorrect dim=1 (True) or correct dim=2 (False)
    """
    # Compute attention scores
    attn = torch.einsum('bhid,bhjd->bhij', q, k)
    
    # The bug: softmax along wrong dimension
    if buggy:
        attn = attn.softmax(dim=1)  # Incorrect (original bug)
    else:
        attn = attn.softmax(dim=2)  # Correct
    
    # Apply attention to values
    out = torch.einsum('bhij,bhjd->bhid', attn, v)
    return out, attn

# Test parameters
batch_size = 2
heads = 3
seq_len = 4
dim = 5

# Create random inputs
q = torch.randn(batch_size, heads, seq_len, dim)
k = torch.randn(batch_size, heads, seq_len, dim)
v = torch.randn(batch_size, heads, seq_len, dim)

# Run both versions
output_buggy, attn_buggy = attention_bug_repro(q, k, v, buggy=True)
output_correct, attn_correct = attention_bug_repro(q, k, v, buggy=False)

# Compare results
print("Attention weights shapes:", attn_buggy.shape, attn_correct.shape)
print("Max difference in outputs:", (output_buggy - output_correct).abs().max().item())

# Verify softmax application
print("\nBuggy attention sums along dim=1 (should not sum to 1):")
print(attn_buggy.sum(dim=1))  # Should NOT be all 1s

print("\nCorrect attention sums along dim=2 (should sum to 1):")
print(attn_correct.sum(dim=2))  # Should be all 1s