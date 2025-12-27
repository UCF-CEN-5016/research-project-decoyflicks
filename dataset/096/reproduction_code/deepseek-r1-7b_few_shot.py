import torch
import random

# Example code that fails with self-conditioning in eval mode:
class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )
        
    def forward(self, x, t):
        return self.net(x * t)  # Simplified model

    def get_model_predictions(self, x, t):
        return self.net(x), None  # Return prediction and None for var_x_prev

# Creates issue when using inference_mode due to gradient attempts
def worker():
    if self.conditioning and random.random() < 0.5:
        with torch.no_grad():  # Wrap in no_grad to prevent gradient tracking issues
            x_self_cond = self.model_predictions(x.clone(), t)  # No need for detach here
            
    model.eval()
    
    x = torch.randn(32, 10)
    t = torch.randn(1, 1).expand(x.size(0), -1)

    with torch.no_grad():
        outputs = self.model(x, t)
        loss = ((outputs - y) ** 2).mean()
        
    print(f"Loss: {loss.item()}")