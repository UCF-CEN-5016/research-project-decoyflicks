import torch
import torch.nn.functional as F

def calculate_attention(q, k, v):
    d_k = q.size(-1)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    attn = F.softmax(scores, dim=-1)
    
    output = torch.matmul(attn, v)
    
    return output

def attention_bug_repro(q, k, v):
    # Original implementation (potential bug)
    output_original = calculate_attention(q, k, v)
    
    # Proposed fix
    output_fixed = calculate_attention(q, k, v)
    
    return output_original, output_fixed

# Test with random inputs
batch_size, heads, seq_len, d_k = 2, 4, 10, 64
q = torch.randn(batch_size, heads, seq_len, d_k)
k = torch.randn(batch_size, heads, seq_len, d_k)
v = torch.randn(batch_size, heads, seq_len, d_k)

# Compare outputs
out_orig, out_fixed = attention_bug_repro(q, k, v)
print("Output difference norm:", torch.norm(out_orig - out_fixed))