import torch
from einops import rearrange
from torch import nn

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, **kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads

    def forward(self, x, mask=None, mems=None, return_mems=False):
        if mems is None:
            mems = [torch.zeros_like(x) for _ in range(self.depth)]
        return x, mems if return_mems else x

class ContinuousTransformerWrapper(nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super().__init__()
        self.attn_layers = attn_layers
        self.dim_out = dim_out
        self.max_mem_len = max_mem_len
        self.num_memory_tokens = num_memory_tokens

    def forward(self, x, mask=None, mems=None, return_mems=False):
        x, mems = self.attn_layers(x, mask=mask, mems=mems, return_mems=return_mems)
        return x, mems if return_mems else x

net = ContinuousTransformerWrapper(
    dim_out=259,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=4,
    )
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]
logits, mems = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)
print([m.shape for m in mems])