import torch
from random import random

# Minimal diffusion model components
class MockModel(torch.nn.Module):
    def forward(self, x, t):
        return x * 0.8  # Simple mock behavior

class DiffusionWrapper:
    def __init__(self, self_condition=True):
        self.model = MockModel()
        self.self_condition = self_condition
    
    def model_predictions(self, x, t):
        return type('obj', (object,), {'pred_x_start': self.model(x, t)})
    
    def forward(self, x, t):
        if self.self_condition and random() < 0.5:
            # This will trigger the error
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
            x = x + x_self_cond  # Later needs grad
        
        return self.model(x, t)

# Trigger the error
model = DiffusionWrapper()
x = torch.randn(1, 3, requires_grad=True)
t = torch.tensor([1])

try:
    out = model.forward(x, t)
    out.sum().backward()  # This will raise RuntimeError
except RuntimeError as e:
    print("Error reproduced:", e)

# Working version with no_grad
print("\nWorking with no_grad:")
class FixedDiffusionWrapper(DiffusionWrapper):
    def forward(self, x, t):
        if self.self_condition and random() < 0.5:
            with torch.no_grad():  # This works
                x_self_cond = self.model_predictions(x.clone().detach(), t).pred_x_start
                x_self_cond = x_self_cond.detach_()
            x = x + x_self_cond
        
        return self.model(x, t)

fixed_model = FixedDiffusionWrapper()
out = fixed_model.forward(x, t)
out.sum().backward()  # Works fine
print("Backward pass completed successfully")