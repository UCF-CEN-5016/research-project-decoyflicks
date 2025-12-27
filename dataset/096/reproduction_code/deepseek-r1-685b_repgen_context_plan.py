import torch
import torch.nn as nn
from types import SimpleNamespace
from random import random

class SelfConditionModel(nn.Module):
    def __init__(self, self_condition=True):
        super(SelfConditionModel, self).__init__()
        self.self_condition = self_condition
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
    
    def model_predictions(self, x):
        return SimpleNamespace(pred_x_start=self.net(x))
    
    def forward(self, x, t):
        x_self_cond = torch.zeros_like(x)
        
        if self.self_condition and random() < 0.5:
            x_self_cond = self.model_predictions(x.clone()).pred_x_start.detach()
        
        return self.net(x + x_self_cond)

# Test case that triggers the error
model = SelfConditionModel()
x = torch.randn(1, 10, requires_grad=True)
t = torch.tensor([0])
out = model(x, t)
loss = out.sum()
loss.backward()  # RuntimeError: Inference tensors cannot be saved for backward

with torch.no_grad():
    x_self_cond = model.model_predictions(x.clone()).pred_x_start.detach()