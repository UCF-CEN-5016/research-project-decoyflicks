import torch
from x_transformers import TransformerWrapper, Decoder

torch.manual_seed(42)

num_tokens = 2049
max_seq_len = 500
model = TransformerWrapper(
    num_tokens=num_tokens,
    max_seq_len=max_seq_len,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
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
context = torch.rand(2, 4, 1024)
context_mask = torch.zeros(2, 4, dtype=torch.bool)

out = model(input_data, context=context, context_mask=context_mask)
assert torch.isnan(out).any()