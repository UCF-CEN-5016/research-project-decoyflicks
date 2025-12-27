import torch
from torch import nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x, training=True):
        # Ensure 'return_loss' is defined before use (removed for this example)
        if training:
            x = self.dropout(x)
        return x