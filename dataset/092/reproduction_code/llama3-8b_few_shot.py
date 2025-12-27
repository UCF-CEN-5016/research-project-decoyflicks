import xtransformers as xt

# Define a transformer wrapper with cross-attention
decoder = xt.TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=xt.Decoder(
        dim=1024,
        depth=24,
        heads=16,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True,
    ),
)

# Generate random input and context
i = torch.randint(0, 2048, (2, 20))
context = torch.rand(2, 4, 1024)
context_mask = torch.zeros(2, 4, dtype=torch.bool)  # All padding

# Run the decoder with all-padded context
out = decoder(i, context=context, context_mask=context_mask)

print("Decoder output:", out)