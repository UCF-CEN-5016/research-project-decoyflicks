import torch
from torch import nn

class DummyAttention(nn.Module):
    def __init__(self, dim_head, heads, kv_heads, qk_norm):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm

        if self.qk_norm:
            # Incorrect shape when kv_heads != heads
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
        else:
            self.qk_norm_k_scale = None

    def forward(self, x):
        # dummy forward to use the parameter
        if self.qk_norm_k_scale is not None:
            return self.qk_norm_k_scale * x
        return x

# Setup with qk_norm=True and kv_heads != heads triggers shape conflict
dim_head = 64
heads = 8
kv_heads = 4
qk_norm = True

model = DummyAttention(dim_head, heads, kv_heads, qk_norm)

x = torch.randn(kv_heads, 1, dim_head)

# This will fail because qk_norm_k_scale shape is (heads,1,dim_head) but input x shape is (kv_heads,1,dim_head)
output = model(x)
print(output.shape)