import torch
from x_transformers import AutoFlashAttn

class Model:
    def __init__(self):
        self.dim_head = 32
        self.kv_enc_dim = 1024
        
        self.attn = AutoFlashAttn(
            query_key_value_norms=(True, False),
            kv_heads=self.kv_enc_dim // self.dim_head,
            qk_norm=True,
            norm_type='ln'
        )

model = Model()

# Set kv_heads to a specific value that causes the issue
model.kv_enc_dim = 1024
model.kv_heads = model.kv_enc_dim // model.dim_head

# Adjust the parameter shape to fix the conflict with qk_norm
model.qk_norm_k_scale = nn.Parameter(torch.ones(model.kv_heads, 1, model.dim_head))

import torch
from x_transformers import AutoFlashAttn
import torch.nn as nn

class MinimalModel:
    def __init__(self):
        self.dim_head = 32
        self.kv_enc_dim = 1024
        
        self.attn = AutoFlashAttn(
            query_key_value_norms=(True, False),
            kv_heads=self.kv_enc_dim // self.dim_head,
            qk_norm=True,
            norm_type='ln'
        )
        
        # Set the parameter shape based on kv_heads
        self.qk_norm_k_scale = nn.Parameter(torch.ones(self.kv_enc_dim // self.dim_head, 1, self.dim_head))

model = MinimalModel()