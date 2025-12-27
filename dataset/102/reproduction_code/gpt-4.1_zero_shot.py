import torch
import torch.nn as nn

class ResidualSimVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True
        self.quantize_dropout = True

    def forward(self, x):
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
        return x

model = ResidualSimVQ()
x = torch.randn(2, 3)
model(x)