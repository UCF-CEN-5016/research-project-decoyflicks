import torch
from torch import nn
from random import random

class DummyModel(nn.Module):
    def forward(self, x, t):
        x.requires_grad_()
        return type('Out', (), {'pred_x_start': x * 2})

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_condition = True
        self.model_predictions = DummyModel()

    def forward(self, x, t):
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
        else:
            x_self_cond = None
        loss = (x * 2).sum()
        loss.backward()

x = torch.randn(2, 2, requires_grad=True)
t = torch.tensor([1,2])
m = MyModule()
m(x, t)