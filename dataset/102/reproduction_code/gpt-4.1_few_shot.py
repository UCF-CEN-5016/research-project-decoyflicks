import torch
import torch.nn as nn

# Minimal stub for ResidualSimVQ with the problematic line
class ResidualSimVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True
        self.quantize_dropout = True

    def forward(self, x):
        # This line triggers NameError because return_loss is not defined anywhere
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
        return should_quantize_dropout

# Instantiate and run forward to reproduce the NameError
model = ResidualSimVQ()
x = torch.randn(2, 3, 4)
model(x)