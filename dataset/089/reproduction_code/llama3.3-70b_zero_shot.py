import torch
import torch.nn as nn

class TestModule(nn.Module):
    def __init__(self, dim, heads, kv_heads, dim_head):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))

    def forward(self):
        return self.qk_norm_k_scale

# set up the model
model = TestModule(dim=128, heads=8, kv_heads=16, dim_head=32)
print(model.forward().shape)

# set up the model with proposed fix
class TestModuleFixed(nn.Module):
    def __init__(self, dim, heads, kv_heads, dim_head):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

    def forward(self):
        return self.qk_norm_k_scale

model_fixed = TestModuleFixed(dim=128, heads=8, kv_heads=16, dim_head=32)
print(model_fixed.forward().shape)