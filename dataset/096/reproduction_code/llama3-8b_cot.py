import torch
import torch.nn as nn
import random

# Minimal environment setup
torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)

    def forward(self, x):
        return torch.relu(self.fc1(x))

model = MyModel().to(device)

# Define the self-conditioning code
class SelfConditioned(nn.Module):
    def __init__(self, model):
        super(SelfConditioned, self).__init__()
        self.model = model

    def forward(self, x, t):
        if random() < 0.5:
            with torch.inference_mode():
                pred_x_start = self.model(x.clone().detach(), t).pred_x_start
                pred_x_start = pred_x_start.detach_()  # This line causes the bug
        else:
            pred_x_start = self.model(x, t).pred_x_start
        return pred_x_start

# Trigger the bug
self_conditioned_model = SelfConditioned(model)
x = torch.randn(1, 10).to(device)
t = torch.tensor([0.]).to(device)

try:
    with torch.no_grad():  # This line fixes the bug
        result = self_conditioned_model(x, t)
except RuntimeError as e:
    print(f"Bug triggered: {e}")