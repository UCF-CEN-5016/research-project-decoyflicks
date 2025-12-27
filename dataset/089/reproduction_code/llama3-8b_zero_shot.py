import torch
from x_transformers import XTransformers, KVHeads

class Model(XTransformers):
    def __init__(self):
        super().__init__()
        self.kv_heads = 2
        self.heads = 1
        self.qk_norm_k_scale = nn.Parameter(torch.randn(1))

model = Model()