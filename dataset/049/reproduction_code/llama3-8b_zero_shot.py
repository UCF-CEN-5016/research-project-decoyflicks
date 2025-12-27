import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mha = nn.MultiHeadAttention(1, 1)

    def forward(self, x):
        return self.mha(x, x)

model = Model()
torch.manual_seed(0)
print(model.forward(torch.randn(1, 10)))