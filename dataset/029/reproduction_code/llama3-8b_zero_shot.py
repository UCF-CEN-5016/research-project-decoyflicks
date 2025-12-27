import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 2)

    def forward(self, x):
        return torch.relu(self.fc1(x))

net = Net()

if torch.backends.mps.is_available():
    print("MPS is available")
else:
    print("MPS is not available")