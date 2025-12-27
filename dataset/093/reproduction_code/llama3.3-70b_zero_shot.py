import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, rotary_pos_emb, attn_flash, attn_onnxable, attn_num_mem_kv, attn_one_kv_head):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.rotary_pos_emb = rotary_pos_emb
        self.attn_flash = attn_flash
        self.attn_onnxable = attn_onnxable
        self.attn_num_mem_kv = attn_num_mem_kv
        self.attn_one_kv_head = attn_one_kv_head

    def forward(self, x):
        if self.attn_num_mem_kv > 0 and self.attn_one_kv_head:
            raise ValueError("attn_num_mem_kv > 0 and attn_one_kv_head = True")
        return torch.randn(x.shape[0], x.shape[1], self.dim)

class TransformerWrapper(nn.Module):
    def __init__(self, num_tokens, max_seq_len, num_memory_tokens, attn_layers):
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens
        self.attn_layers = attn_layers

    def forward(self, x):
        return self.attn_layers(x)

lm = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    num_memory_tokens=20,
    attn_layers=Decoder(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=20,
        attn_one_kv_head=True
    )
)

x = torch.randint(0, 32, (8, 1024))
try:
    logits = lm(x)
    print(x.shape, x.dtype)
    print(logits.shape, logits.dtype)
except ValueError as e:
    print(e)