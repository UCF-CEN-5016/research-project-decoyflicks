import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.no_grad import inference_mode

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
        
        if self.self_condition and random() < 0.5:
            with inference_mode():
                x_self_cond = self.model_predictions(x.clone(), t).pred_x_start.detach()
        
        return self.net(x + x_self_cond)

# Test case without triggering the error
model = SelfConditionModel()
x = torch.randn(1, 10, requires_grad=True)
t = torch.tensor([0])
out = model(x, t)
loss = out.sum()
loss.backward()