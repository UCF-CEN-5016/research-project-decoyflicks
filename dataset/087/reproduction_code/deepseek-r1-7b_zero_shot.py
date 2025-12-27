x = torch.mm(x, x.T)

import torch
from x_transformers.positional import (
    apply_alibi,
    FlashAttentionBias,
)

# Function that computes position alibi-based attention bias for flash attention
def apply_alibi_pos(x, mask=None):
    # Compute all possible position differences
    seq_len = x.shape[1]
    pos_i = torch.arange(seq_len, device=x.device).view(-1, 1)
    pos_j = torch.arange(seq_len, device=x.device).view(1, -1)
    distance = pos_j - pos_i

    # Compute alibi attention bias
    with torch.no_grad():
        x = apply_alibi(x, mask=mask)
    
    if isinstance(x, FlashAttentionBias):
        if x.flash:
            x = x.attn.causal
        else:
            x = (
                (x.attn present) and x.attn.bias
            )
            
    # Add an extra dimension to handle 4D tensors for flash attention
    x = x.unsqueeze(2)
    
    # Create lower triangular mask of ones
    mask = torch.tril(torch.ones(seq_len, seq_len))
    if x.dim() == 3:
        mask = mask.unsqueeze(0)
    elif x.dim() == 4:
        mask = mask.unsqueeze(0).unsqueeze(2)
        
    x = x * mask
    
    return x