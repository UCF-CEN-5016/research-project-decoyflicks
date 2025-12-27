import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.attention = nn.MultiHeadAttention(1, 1)

    def forward(self, input_tensor):
        return self.attention(input_tensor, input_tensor)

model = SimpleModel()
torch.manual_seed(0)
print(model.forward(torch.randn(1, 10)))