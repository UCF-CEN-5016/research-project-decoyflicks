import torch.nn.functional as F
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 3)

    def forward(self, x):
        return F.gelu(x, approximate=True)

model = MyModel()
input_data = torch.randn(1, 10)
output = model(input_data)