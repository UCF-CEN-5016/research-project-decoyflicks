import torch
import torch.nn.functional as F

def attention_bug_repro(q, k, v):
    """Reproduce the softmax axis bug in self-attention"""
    # Original implementation (potential bug)
    d_k = q.size(-1)
    scores = torch.einsum('bhid,bhjd->bhij', q, k) / torch.sqrt(torch.tensor(d_k))
    attn_original = F.softmax(scores, dim=-1)  # Current implementation (dim=-1)
    output_original = torch.einsum('bhij,bhjd->bhid', attn_original, v)
    
    # Proposed fix
    scores_fixed = torch.einsum('bhid,bhjd->bhij', q, k) / torch.sqrt(torch.tensor(d_k))
    attn_fixed = F.softmax(scores_fixed, dim=2)  # Should be along j dimension (dim=2)
    output_fixed = torch.einsum('bhij,bhjd->bhid', attn_fixed, v)
    
    return output_original, output_fixed

# Test with random inputs
batch_size, heads, seq_len, d_k = 2, 4, 10, 64
q = torch.randn(batch_size, heads, seq_len, d_k)
k = torch.randn(batch_size, heads, seq_len, d_k)
v = torch.randn(batch_size, heads, seq_len, d_k)

# Compare outputs
out_orig, out_fixed = attention_bug_repro(q, k, v)
print("Output difference norm:", torch.norm(out_orig - out_fixed))
print("Original attention sum along dim 2:", attn_original.sum(dim=2))
print("Fixed attention sum along dim 2:", attn_fixed.sum(dim=2))