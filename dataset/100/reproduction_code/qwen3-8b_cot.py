import torch
from accelerate import Accelerator

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = torch.randn(10, requires_grad=True)
        self.cached_freqs = None  # Caching mechanism

    def forward(self, t):
        if self.cached_freqs is None:
            self.cached_freqs = self.freqs  # Cache the tensor
        return self.cached_freqs

# Initialize accelerator and model
accelerator = Accelerator()
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy input
input_tensor = torch.randn(5, 10)

# Training loop
for step in range(2):
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()  # First backward call
    optimizer.step()
    optimizer.zero_grad()

    # Second backward call (will trigger error)
    loss = output.sum()
    loss.backward()

loss.backward(retain_graph=True)