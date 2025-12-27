import torch
from x_transformers import TransformerWrapper

# Define decoder with cross attention
decoder = TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=TransformerWrapper_ATtnLayers(
        dim=1024,
        depth=24,
        heads=16,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True,
    )
)

class TransformerWrapper_ATtnLayers:
    def __init__(self, dim, depth, heads, ...):
        # Initialize transformer attention layers
        pass

# Prepare input tokens (minimized example)
batch_size = 2
seq_len_input = 5  # Simplified from 20 for clarity
i = torch.randint(0, 2048, (batch_size, seq_len_input))

# Prepare context and mask with all padding (mask is False everywhere)
seq_len_context = 4
dim_context = 1024
context = torch.randn(batch_size, seq_len_context, dim_context)
context_mask = torch.zeros(batch_size, seq_len_context, dtype=torch.bool)

# Get decoder output which should contain NaNs due to all padding context
outputs = decoder(i, context=context, context_mask=context_mask)

print(f"Decoder outputs with all padding context: {outputs}")