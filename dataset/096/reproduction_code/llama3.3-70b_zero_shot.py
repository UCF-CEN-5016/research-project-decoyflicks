import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return {'pred_x_start': x}

class SelfCondition:
    def __init__(self, model):
        self.model = model
        self.self_condition = True

    def model_predictions(self, x, t):
        return self.model(x)

    def run(self, x, t):
        if self.self_condition and random.random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
            return x_self_cond
        else:
            return self.model(x).pred_x_start

model = Model()
self_condition = SelfCondition(model)
x = torch.randn(1, 10)
t = 0
try:
    self_condition.run(x, t)
except RuntimeError as e:
    print(e)

self_condition.self_condition = True
try:
    with torch.no_grad():
        x_self_cond = self_condition.model_predictions(x.clone().detach(), t).pred_x_start
    print(x_self_cond)
except RuntimeError as e:
    print(e)