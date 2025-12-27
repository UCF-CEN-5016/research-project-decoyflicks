import torch
from accelerate import Accelerator

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = torch.randn(10, requires_grad=True)
        self.cached_freqs = None

    def forward(self, t):
        if self.cached_freqs is None:
            self.cached_freqs = self.freqs.clone().detach()
        return self.cached_freqs

def train_step(model, input_tensor, optimizer):
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Initialize accelerator and model
accelerator = Accelerator()
model = MyModel().to(accelerator.device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy input
input_tensor = torch.randn(5, 10).to(accelerator.device)

# Training loop
for step in range(2):
    train_step(model, input_tensor, optimizer)

    # Second backward call (will not trigger error due to detached tensor)
    loss = model(input_tensor).sum()
    loss.backward(retain_graph=True)