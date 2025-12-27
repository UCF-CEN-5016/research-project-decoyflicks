import torch
import torch.nn as nn

class ROPE(nn.Module):
    def __init__(self, cos_cached, sin_cached):
        super(ROPE, self).__init__()
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def forward(self, x):
        x_rope = x * self.cos_cached[:x.shape[0]] + (x[:, 1:] * -0.5) * self.sin_cached[:x.shape[0]]
        return x_rope

cos_cached = torch.tensor([[1., 0.5], [0.5, 1.]])
sin_cached = torch.tensor([[1., 0.5], [0.5, 1.]])
rope = ROPE(cos_cached, sin_cached)

x = torch.tensor([[1., 2], [3., 4]])
x_rope = rope.forward(x)