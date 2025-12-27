import torch
from einops import rearrange
from x_transformers import Encoder

torch.manual_seed(42)

batch_size = 1
input_seq_length = 64
context_seq_length = 128

x = torch.randn((batch_size, input_seq_length, 256))
mask = torch.ones((batch_size, input_seq_length)).bool()
context = torch.randn((batch_size, context_seq_length, 512))
context_mask = torch.ones((batch_size, context_seq_length)).bool()

model = Encoder(
    dim=256,
    depth=4,
    heads=4,
    rotary_pos_emb=True,
    cross_attend=True,
    cross_attn_dim_context=512
)

pos = torch.arange(input_seq_length)
context_pos = torch.arange(context_seq_length)

try:
    embed = model(
        x=x,
        mask=mask,
        context=context,
        pos=pos,
        context_pos=context_pos,
        context_mask=context_mask
    )
except Exception as e:
    if isinstance(e, rearrange.EinopsError) and "Error while processing rearrange-reduction pattern \"n -> n 1\"" in str(e):
        assert str(e).count("Input tensor shape: torch.Size([1, 32])") > 0