import torch
from x_transformers import TransformerWrapper, Decoder

# Create a decoder with cross-attention
d = TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=Decoder(
        dim=1024,
        depth=24,
        heads=16,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True,
    ),
)

# Generate input and context with all padding
i = torch.randint(0, 2048, (2, 20))  # Input tokens
context = torch.rand(2, 4, 1024)  # Random context (all tokens are considered padding)
context_mask = torch.zeros(2, 4, dtype=torch.bool)  # All context is padding

# Call the decoder with all padding context
out = d(i, context=context, context_mask=context_mask)

print("Decoder output:", out)