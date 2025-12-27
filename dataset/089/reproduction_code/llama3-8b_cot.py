import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim_head):
        super(MyModel, self).__init__()
        self.qk_norm = nn.Parameter(torch.ones(1, 1, dim_head))
        self.kv_heads = nn.Parameter(torch.ones(2, 1, dim_head))  # kv_heads != heads
        self.heads = nn.Parameter(torch.ones(1, 1, dim_head))

    def forward(self):
        return self.qk_norm * self.kv_heads

model = MyModel(8)  # Set a specific value for `dim_head`

# Wrap the code in this condition to trigger the bug
if model.kv_heads.shape[0] != model.heads.shape[0]:
    model.qk_norm_k_scale = nn.Parameter(torch.ones(model.kv_heads.shape[0], 1, model.dim_head))