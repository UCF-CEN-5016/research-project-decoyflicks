import torch
import x_transformers as xt

torch.manual_seed(42)

batch_size = 2
num_tokens = 2049
max_seq_len = 500
dim = 1024
depth = 24
heads = 16
attn_dim_head = 64

decoder = xt.TransformerWrapper(
    num_tokens=num_tokens,
    max_seq_len=max_seq_len,
    attn_layers=xt.Decoder(dim=dim, depth=depth, heads=heads, causal=True, cross_attend=True)
)

input_data = torch.randint(0, 2048, (batch_size, 20))
context = torch.rand(batch_size, 4, dim)
context_mask = torch.zeros(batch_size, 4, dtype=torch.bool)

out = decoder(input_data, context=context, context_mask=context_mask)

if torch.isnan(out).any():
    print("NaN values found in the output logits")