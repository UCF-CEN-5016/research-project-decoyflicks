import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def model_predictions(self, x, t):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(32, 10)
t = torch.randint(0, 100, (32,))

# This will raise an error if any gradient operations are performed inside inference mode
with torch.inference_mode():
    pred = model.model_predictions(x.clone().detach(), t)
    # Suppose we accidentally try to do something like:
    # pred = pred + 1  # This would not cause an error, but if we do:
    pred.backward()  # This will throw an error

import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def model_predictions(self, x, t):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(32, 10)
t = torch.randint(0, 100, (32,))

# Use torch.no_grad() instead of torch.inference_mode()
with torch.no_grad():
    pred = model.model_predictions(x.clone().detach(), t)
    # You can now perform operations that do not require gradients
    # For example, you could do:
    # loss = some_loss_function(pred, target)
    # ...