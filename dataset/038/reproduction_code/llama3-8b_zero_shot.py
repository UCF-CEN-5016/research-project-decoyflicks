import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cos_cached = torch.randn(1, 1, 1, 4)
        self.sin_cached = torch.randn(1, 1, 1, 4)

    def forward(self, x):
        x_rope = (x * self.cos_cached[:x.shape[0], :, :, :]) + ((-0.5) * x * self.sin_cached[:x.shape[0], :, :, :])
        return x_rope

model = Model()
x = torch.randn(1, 1, 1, 3)
result = model(x)