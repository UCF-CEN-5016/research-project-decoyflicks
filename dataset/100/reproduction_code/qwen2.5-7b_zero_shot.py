import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([1.0, 2.0], requires_grad=True)
        self.cached_weight = None

    def forward(self, x):
        if self.cached_weight is None:
            self.update_cached_weight()
        return self.cached_weight * x

    def update_cached_weight(self):
        self.cached_weight = self.weight.clone().detach()

# Test the model
model = Model()
x = torch.tensor([3.0, 4.0], requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()