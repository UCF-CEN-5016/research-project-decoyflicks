import torch
from x_transformers import TransformerWrapper, Decoder

torch.manual_seed(42)

batch_size = 2
input_seq_length = 20
context_seq_length = 4
embedding_dim = 1024

i = torch.randint(0, 2048, (batch_size, input_seq_length))
context = torch.rand(batch_size, context_seq_length, embedding_dim)
context_mask = torch.zeros(batch_size, context_seq_length, dtype=torch.bool)

decoder = TransformerWrapper(
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
        cross_attend=True
    )
)

out = decoder(i, context=context, context_mask=context_mask)

assert torch.isnan(out).any()