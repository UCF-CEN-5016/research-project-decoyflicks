import torch
import x_transformers as xt

# Setup decoder with cross-attention enabled
decoder = xt.TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=xt.Decoder(
        dim=1024,
        depth=1,  # minimal depth for repro
        heads=4,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True,
    ),
)

# Input tokens
i = torch.randint(0, 2048, (2, 20))

# Context is non-empty but all masked out (all padding)
context = torch.rand(2, 4, 1024)
context_mask = torch.zeros(2, 4, dtype=torch.bool)  # all False mask

# Forward pass: outputs NaN logits due to empty context in cross-attention
out = decoder(i, context=context, context_mask=context_mask)

print("Output contains NaNs:", torch.isnan(out).any().item())