import torch
from x_transformers import TransformerWrapper
from x_transformers import Decoder  # Fixed the undefined variable issue

torch.manual_seed(42)

num_tokens = 2049
max_seq_len = 500
decoder = TransformerWrapper(
    num_tokens=num_tokens,
    max_seq_len=max_seq_len,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=True,
    attn_layers=Decoder(dim=1024, depth=24, heads=16, attn_dim_head=64, attn_flash=True, ff_no_bias=True, cross_attend=True)
)

batch_size = 2
input_seq_len = 20
# Create a random input tensor
i = torch.randint(0, 2048, (batch_size, input_seq_len))
# Create a random context tensor
context = torch.rand(batch_size, 4, 1024)
# Create a context mask filled with False (indicating all padding)
context_mask = torch.zeros(batch_size, 4, dtype=torch.bool)

# Call the decoder with the input tensor, context tensor, and context mask
out = decoder(i, context=context, context_mask=context_mask)

# Assert that the output contains NaN values, reproducing the bug
assert torch.isnan(out).any()