import torch
from einops import rearrange
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_mem_kv=0, one_kv_head=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.num_mem_kv = num_mem_kv
        self.one_kv_head = one_kv_head

        kv_heads = 1 if one_kv_head else heads
        kv_dim = dim_head * kv_heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(num_mem_kv, dim_head * kv_heads))
            self.mem_v = nn.Parameter(torch.randn(num_mem_kv, dim_head * kv_heads))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        if self.num_mem_kv > 0:
            mem_k = self.mem_k.expand(b, -1, -1)
            mem_v = self.mem_v.expand(b, -1, -1)
            k = torch.cat((mem_k, k), dim=1)
            v = torch.cat((mem_v, v), dim=1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=1 if self.one_kv_head else h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=1 if self.one_kv_head else h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, attn_num_mem_kv=0, attn_one_kv_head=False):
        super().__init__()
        self.layers = nn.ModuleList([
            Attention(dim, heads, dim//heads, attn_num_mem_kv, attn_one_kv_head)
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return x

class TransformerWrapper(nn.Module):
    def __init__(self, num_tokens, max_seq_len, num_memory_tokens, attn_layers):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, attn_layers.layers[0].dim)
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, attn_layers.layers[0].dim))
        self.attn_layers = attn_layers

    def forward(self, x):
        x = self.token_emb(x)
        b, n, d = x.shape
        memory_tokens = self.memory_tokens.expand(b, -1, -1)
        x = torch.cat((memory_tokens, x), dim=1)
        return self.attn_layers(x)

lm = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=20,
    attn_layers=Decoder(
        dim=512,
        depth=4,
        heads=4,
        attn_num_mem_kv=20,
        attn_one_kv_head=True
    )
)

x = torch.randint(0, 32, (8, 1024))
logits = lm(x)