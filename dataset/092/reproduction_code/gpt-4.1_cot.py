import torch
import x_transformers as xt

# Set deterministic seed for reproducibility (optional)
torch.manual_seed(0)

# Instantiate the decoder with cross-attention enabled
decoder = xt.TransformerWrapper(
    num_tokens=2049,
    max_seq_len=500,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=xt.Decoder(
        dim=1024,
        depth=2,          # smaller depth for faster testing
        heads=8,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True,
    ),
)

# Input token ids (batch_size=2, seq_len=20)
i = torch.randint(0, 2048, (2, 20))

# Cross-attention context tensor (batch_size=2, context_len=4, dim=1024)
context = torch.rand(2, 4, 1024)

# Context mask all False (all padding -> no valid context tokens)
context_mask = torch.zeros(2, 4, dtype=torch.bool)

# Forward pass
out = decoder(i, context=context, context_mask=context_mask)

print("Output logits:", out)
print("Are any outputs NaN?", torch.isnan(out).any().item())