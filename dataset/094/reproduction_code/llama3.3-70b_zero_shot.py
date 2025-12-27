import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, rotary_pos_emb, shift_tokens, attn_flash, attn_onnxable, use_rmsnorm, sandwich_norm):
        super().__init__()
        self.layers = nn.ModuleList([self._make_layer(dim, heads, rotary_pos_emb, shift_tokens, attn_flash, attn_onnxable, use_rmsnorm, sandwich_norm) for _ in range(depth)])

    def _make_layer(self, dim, heads, rotary_pos_emb, shift_tokens, attn_flash, attn_onnxable, use_rmsnorm, sandwich_norm):
        return nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim, dropout=0.1, activation='relu')

    def forward(self, x, mask=None, mems=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return x

class ContinuousTransformerWrapper(nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super().__init__()
        self.dim_out = dim_out
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers
        self.fc = nn.Linear(512, dim_out)

    def forward(self, x, mask=None, mems=None, return_mems=False):
        x = self.attn_layers(x, mask=mask, mems=mems)
        x = self.fc(x)
        if return_mems:
            return x, mems
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