import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cos_cached = torch.randn(1, 1, 1, 4)
        self.sin_cached = torch.randn(1, 1, 1, 4)

    def forward(self, x):
        x_rope = x
        neg_half_x = (-0.5) * x
        # Intentional incorrect slicing that can cause a size mismatch when last dims differ
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

model = Model()
x = torch.randn(1, 1, 1, 3)
result = model(x)