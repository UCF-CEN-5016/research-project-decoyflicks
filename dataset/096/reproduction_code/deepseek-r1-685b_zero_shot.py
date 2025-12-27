import torch
import torch.nn as nn
from random import random

class Model(nn.Module):
    def __init__(self, self_condition=True):
        super().__init__()
        self.self_condition = self_condition
        self.layer = nn.Linear(10, 10)
    
    def model_predictions(self, x, t):
        return type('obj', (object,), {'pred_x_start': self.layer(x)})
    
    def forward(self, x, t):
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
            x = x + x_self_cond
        return self.layer(x)

model = Model()
x = torch.randn(1, 10, requires_grad=True)
t = torch.tensor([0])
out = model(x, t)
out.sum().backward()