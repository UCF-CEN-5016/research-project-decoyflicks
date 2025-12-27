import torch
from x_transformers import TransformerWrapper, Decoder

torch.manual_seed(42)

num_tokens = 2049
max_seq_len = 500
decoder = TransformerWrapper(
    num_tokens=num_tokens,
    max_seq_len=max_seq_len,
    attn_layers=Decoder(
        dim=1024,
        depth=24,
        heads=16,
        attn_dim_head=64,
        attn_flash=True,
        ff_no_bias=True,
        cross_attend=True
    )
)

batch_size = 2
input_seq_len = 20
input_data = torch.randint(0, 2048, (batch_size, input_seq_len))
context = torch.rand(batch_size, 4, 1024)
context_mask = torch.zeros(batch_size, 4, dtype=torch.bool)

out = decoder(input_data, context=context, context_mask=context_mask)
assert torch.isnan(out).any()