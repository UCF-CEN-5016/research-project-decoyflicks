import torch
from your_module import ContinuousTransformerWrapper, Decoder

dim_out = 259
max_seq_len = 0
num_memory_tokens = 20
max_mem_len = 100
attn_layers = Decoder(
    dim=512,
    depth=6,
    heads=4,
    rotary_pos_emb=True,
    shift_tokens=1,
    attn_flash=True,
    attn_onnxable=True,
    use_rmsnorm=True,
    sandwich_norm=True
)

net = ContinuousTransformerWrapper(
    dim_out=dim_out,
    max_seq_len=max_seq_len,
    num_memory_tokens=num_memory_tokens,
    max_mem_len=max_mem_len,
    attn_layers=attn_layers
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]

logits, mems_out = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)
print([mm.shape for mm in mems_out])