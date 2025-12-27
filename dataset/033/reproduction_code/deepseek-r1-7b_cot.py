import torch
from torch import nn

class SE3TransformerPooled(nn.Module):
    def __init__(self, fiber_in, fiber_out, hidden_size=64):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(fiber_in, 8)
        # Updated attention layers to handle variable dimensions
        self.slf_attn_k = nn.Linear(fiber_in, hidden_size)
        self.slf_attn_v = nn.Linear(fiber_in, hidden_size)
        
    def forward(self, x):
        # Process each node type separately based on their features
        out_0 = self.slf_attn(x['0'])  # handles feature dim 5
        out_1 = self.slf_attn(x['1'])  # handles feature dim 1
        
        # Concatenate outputs and process further
        pooled_out = torch.cat([out_0, out_1], dim=0)
        return self.slf_attn_pooled(pooled_out)