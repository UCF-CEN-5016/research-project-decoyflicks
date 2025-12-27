import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def model_predictions(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(32, 10)
t = torch.randint(0, 100, (32,))

with torch.no_grad():
    pred = model.model_predictions(x.clone().detach())
    # You can now perform operations that do not require gradients
    # For example, you could do:
    # loss = some_loss_function(pred, target)