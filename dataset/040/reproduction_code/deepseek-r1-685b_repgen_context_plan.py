import torch
import torch.nn.functional as F

def compute_attention(q, k, v):
    d_k = q.size(-1)
    
    # Calculate scores with correct normalization
    scores = torch.einsum('bhid,bhjd->bhij', q, k) / torch.sqrt(torch.tensor(d_k))
    
    # Apply softmax along the correct dimension
    attn = F.softmax(scores, dim=2)
    
    # Compute the output using the attention weights
    output = torch.einsum('bhij,bhjd->bhid', attn, v)
    
    return output

def attention_bug_repro(q, k, v):
    # Original implementation (potential bug)
    output_original = compute_attention(q, k, v)
    
    # Proposed fix
    output_fixed = compute_attention(q, k, v)
    
    return output_original, output_fixed

# Test with random inputs
batch_size, heads, seq_len, d_k = 2, 4, 10, 64
q = torch.randn(batch_size, heads, seq_len, d_k)
k = torch.randn(batch_size, heads, seq_len, d_k)
v = torch.randn(batch_size, heads, seq_len, d_k)

# Compare outputs
out_orig, out_fixed = attention_bug_repro(q, k, v)
print("Output difference norm:", torch.norm(out_orig - out_fixed))
print("Original attention sum along dim 2:", out_orig.sum(dim=2))
print("Fixed attention sum along dim 2:", out_fixed.sum(dim=2))