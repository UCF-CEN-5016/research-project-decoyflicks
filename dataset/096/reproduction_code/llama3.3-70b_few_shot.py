import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

class SelfConditioningModel:
    def __init__(self, model):
        self.model = model
        self.self_condition = True

    def model_predictions(self, x, t):
        return self.model(x)

    def forward(self, x, t):
        if self.self_condition and torch.rand(1) < 0.5:
            # Using inference mode
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x.clone().detach(), t)
                x_self_cond = x_self_cond.detach_()
                # This will cause RuntimeError: Inference tensors cannot be saved for backward
                loss = F.mse_loss(x_self_cond, x)
                loss.backward()
        return x

# Initialize model and self-conditioning model
model = Model()
self_conditioning_model = SelfConditioningModel(model)

# Sample data
x = torch.randn(1, 10)
t = 0

# This will cause RuntimeError
self_conditioning_model.forward(x, t)