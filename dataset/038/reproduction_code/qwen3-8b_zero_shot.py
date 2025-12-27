import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_cached = torch.randn(10, 2, 4)
        self.sin_cached = torch.randn(10, 2, 4)

    def forward(self, x):
        seq_len = x.shape[0]
        x_rope = x
        neg_half_x = -x
        x_rope = (x_rope * self.cos_cached[:seq_len]) + (neg_half_x * self.sin_cached[:seq_len])
        return x_rope

model = RotaryEmbedding()
x = torch.randn(5, 2, 3)
output = model(x)
print(output)