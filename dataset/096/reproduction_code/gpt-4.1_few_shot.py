import torch
from random import random

class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        # Simulate model output with a tensor requiring grad
        pred_x_start = x * 2  # simple operation
        return type('Output', (), {'pred_x_start': pred_x_start})

class SelfConditioningModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_predictions = DummyModel()
        self.self_condition = True

    def forward(self, x, t):
        if self.self_condition and random() < 1.0:
            # This causes RuntimeError during backward:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
            # Using x_self_cond in subsequent computation that requires grad
            out = x + x_self_cond
        else:
            out = x * 3
        return out

model = SelfConditioningModule()
x = torch.randn(4, 4, requires_grad=True)
t = torch.tensor([1])

# Forward pass
out = model(x, t)

# Backward pass - will raise RuntimeError if inference_mode used above
out.sum().backward()
print("Backward pass completed successfully")