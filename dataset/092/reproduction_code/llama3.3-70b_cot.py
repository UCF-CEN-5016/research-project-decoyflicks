import torch
import x_transformers as xt

# Minimal environment setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
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
).to(device)

# Triggering conditions
input_ids = torch.randint(0, 2048, (2, 20)).to(device)
context = torch.rand(2, 4, 1024).to(device)
context_mask = torch.zeros(2, 4, dtype=torch.bool).to(device)

# Reproduce the bug
out = decoder(input_ids, context=context, context_mask=context_mask)

# Check if the output is all NaN
print(torch.isnan(out).all())