import torch
import random

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Linear(10, 1)

    def forward(self, x, t):
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
        return x + x_self_cond

model = Model()
x = torch.randn(1, 10)
t = 0

try:
    model(x, t)
except RuntimeError as e:
    print(f"Error: {e}")