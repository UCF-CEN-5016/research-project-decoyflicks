import torch
from x_transformers import TransformerWrapper, DefaultAttention

class CustomPositionAlibi(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.randn(x.shape[0], x.shape[1], self.dim)

model = TransformerWrapper(
    num_tokens=100,
    max_seq_len=100,
    attn_dim=128,
    dim=128,
    depth=1,
    heads=8,
    attn_layer=DefaultAttention,
    alibi=None,
    pos_emb=None,
    attn_flash=True
)

alibi = CustomPositionAlibi(dim=128)
model.attn.alibi = alibi

x = torch.randint(0, 100, (1, 10))
try:
    model(x)
except Exception as e:
    print(e)