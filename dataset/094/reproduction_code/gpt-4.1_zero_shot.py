import torch
from torch import nn

class DecoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x, mem=None):
        q = k = x if mem is None else torch.cat([mem, x], dim=1)
        attn_out, _ = self.attn(x.transpose(0,1), k.transpose(0,1), v=k.transpose(0,1))
        x = x + attn_out.transpose(0,1)
        x = self.norm(x)
        x = x + self.ff(x)
        return x

class ContinuousTransformerWrapper(nn.Module):
    def __init__(self, dim_out, num_memory_tokens, max_mem_len, attn_layers):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.layers = attn_layers
        self.proj = nn.Linear(512, dim_out)

    def forward(self, x, mask=None, mems=None, return_mems=False):
        if mems is None:
            mems = [torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype) for _ in range(len(self.layers))]
        for i, layer in enumerate(self.layers):
            x = layer(x, mems[i])
        logits = self.proj(x)
        if return_mems:
            new_mems = [x[:, -self.max_mem_len:] for _ in range(len(self.layers))]
            return logits, new_mems
        return logits

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, rotary_pos_emb, shift_tokens, attn_flash, attn_onnxable, use_rmsnorm, sandwich_norm):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim) for _ in range(depth)])

    def forward(self, x, mems=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, mems[i] if mems is not None else None)
        return x

net = ContinuousTransformerWrapper(
    dim_out=259,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    attn_layers=Decoder(
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
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]
logits, mems = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)
print([m.shape for m in mems])