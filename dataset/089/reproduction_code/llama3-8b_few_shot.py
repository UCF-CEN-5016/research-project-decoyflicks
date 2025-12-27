import torch
from x_transformers import XFormer

# Create a model with conflicting settings
model = XFormer(dim_head=8, kv_heads=16)  # KV heads different from heads
model.qk_norm_k_scale = nn.Parameter(torch.ones(model.kv_heads, 1, model.dim_head))  # Enable qk_norm

# Sample input
input_ids = torch.tensor([[1, 2, 3]])
attention_mask = torch.tensor([[1, 1, 1]])

# Reproduce the bug
output = model(input_ids, attention_mask)

print("Output:", output)