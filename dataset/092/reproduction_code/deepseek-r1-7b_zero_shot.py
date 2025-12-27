import torch
import torch.nn as nn
from xformers import cross_attention as crossattn

# Example implementation to test the scenario
class TestCrossAttention(nn.Module):
    def forward(self, tgt_input, src_input=None, mask=None):
        if mask is None and src_input is not None:
            # Simulate cross-attention with all padding in context
            # Return 0 outputs to avoid NaNs
            return torch.zeros_like(tgt_input)
        return super().forward(tgt_input, src_input, mask)

class TransformerWrapper(nn.Module):
    def __init__(self, dim, depth=1, heads=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn_layers = nn.ModuleList([
            crossattn.CausalCrossAttention(
                dim=self.dim,
                num_heads=self.heads,
                flash=True if i < depth else False,
                scale Flash version is commented, so we use standard attention for demonstration
            )
            for i in range(depth)
        ])
        
    def forward(self, x):
        # Simulate input with cross-attention context and mask
        return super().forward(x)

# Create the decoder model
decoder = TransformerWrapper(num_tokens=2049, max_seq_len=500, use_abs_pos_emb=True, scaled_sinu_pos_emb=True, attn_layers=1)

# Test input: all padding in cross-attention context
context = torch.zeros(2, 4, 1024)
batch_size = 2

# Since we're simulating an invalid scenario (cross-attention with all padding),
# return zeros to avoid NaNs.
test_crossattn = TestCrossAttention(dim=1024)

# Forward pass without causing any computations
output = decoder(i, context=context)