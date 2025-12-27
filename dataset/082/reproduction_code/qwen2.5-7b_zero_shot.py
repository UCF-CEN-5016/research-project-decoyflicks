import torch
import torch.nn as nn

class MPP(nn.Module):
    def __init__(self, dim):
        super(MPP, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(32 * 32, 3072)  # Output size is 3072
        self.norm = nn.LayerNorm(self.dim)      # Expected input size is 1024

    def forward(self, x):
        x = x.view(x.size(0), -1)               # Reshape x to [batch, 32*32]
        x = self.linear(x)                      # x now has shape [batch, 3072]
        x = x.view(x.size(0), 64, -1)           # Reshape x to [batch, 64, 3072]
        x = x.transpose(1, 2)                   # Transpose to [batch, 3072, 64] for LayerNorm
        x = self.norm(x)                        # LayerNorm expects input of size 1024
        x = x.transpose(1, 2)                   # Transpose back to [batch, 64, 3072]
        return x

# Create a dummy input with shape [batch, 64, 32*32] = [20, 64, 1024]
input_tensor = torch.randn(20, 64, 32 * 32)

# Instantiate the model
model = MPP(dim=1024)

# Forward pass
output = model(input_tensor)