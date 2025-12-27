import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        # Call directly through torch.nn.functional.gelu with a boolean to reproduce the TypeError in some PyTorch versions
        return torch.nn.functional.gelu(x, approximate=True)

model = MyModel()
input_data = torch.randn(1, 10)
output = model(input_data)
print(output)