import torch
import torch.nn as nn
from random import random

class SelfConditionModel(nn.Module):
    def __init__(self, self_condition=True):
        super().__init__()
        self.self_condition = self_condition
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
    
    def model_predictions(self, x, t):
        return SimpleNamespace(pred_x_start=self.net(x))
    
    def forward(self, x, t):
        x_self_cond = torch.zeros_like(x)
        
        # Problematic inference mode block
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():  # Causes RuntimeError
                x_self_cond = self.model_predictions(x.clone(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
        
        return self.net(x + x_self_cond)

# Test case that triggers the error
model = SelfConditionModel()
x = torch.randn(1, 10, requires_grad=True)
t = torch.tensor([0])
out = model(x, t)
loss = out.sum()
loss.backward()  # RuntimeError: Inference tensors cannot be saved for backward

with torch.no_grad():  # Instead of inference_mode()
    x_self_cond = self.model_predictions(x.clone(), t).pred_x_start
    x_self_cond = x_self_cond.detach_()