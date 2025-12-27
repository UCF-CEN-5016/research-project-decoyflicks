import torch
from torch import nn

class DummyModel(nn.Module):
    def __init__(self, dim_head, heads, kv_heads, qk_norm):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.qk_norm = qk_norm

        if qk_norm:
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            # The bug triggers here when kv_heads != heads and qk_norm is True
            # The fix would be using kv_heads instead of heads in the above line

    def forward(self, x):
        return self.qk_norm_k_scale

model = DummyModel(dim_head=4, heads=2, kv_heads=1, qk_norm=True)
print(model())