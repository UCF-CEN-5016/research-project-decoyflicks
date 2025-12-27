import torch
from torch.utils.checkpoint import checkpoint

class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        processed = torch.nested.map(self.linear, x)
        return processed

model = NestedModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

nested_input = torch.nested.nested_tensor([
    torch.randn(5, 1024),
    torch.randn(5, 1024)
])

output = model(nested_input)
loss = output.sum()
loss.backward()