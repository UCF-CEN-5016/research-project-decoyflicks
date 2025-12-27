import torch
import torch.nn as nn
import random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 20)

    def forward(self, x):
        return torch.relu(self.fc1(x))

model = Model()
x = torch.randn(1, 10)
t = torch.tensor([0])

def test():
    if True:  # replace with your condition
        with torch.inference_mode():
            x_self_cond = model(model.predictions(x.clone().detach(), t).pred_x_start).detach_()
test()