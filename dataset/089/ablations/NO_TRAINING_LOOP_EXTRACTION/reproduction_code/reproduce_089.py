import torch
import torch.nn as nn
from x_transformers.x_transformers import XTransformer

kv_heads = 4
heads = 8
dim_head = 64

class TestXTransformer(XTransformer):
    def __init__(self, qk_norm=True):
        super().__init__()
        self.qk_norm = qk_norm
        self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

    def forward(self, x):
        return x @ self.qk_norm_k_scale

batch_size = 2
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, 512)

model = TestXTransformer()
output1 = model(input_tensor)

model.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
output2 = model(input_tensor)

print("Output 1 shape:", output1.shape)
print("Output 2 shape:", output2.shape)
print("Outputs are equal:", torch.equal(output1, output2))