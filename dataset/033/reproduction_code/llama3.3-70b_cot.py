import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

# Define a simple SE3-Transformer layer
class ConvSE3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSE3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, edge_index):
        # Simplified forward pass to demonstrate the bug
        x = torch.matmul(x, self.weight)
        return x

# Define a simple SE3-Transformer model
class SE3TransformerPooled(nn.Module):
    def __init__(self, fiber_in, fiber_out, output_dim):
        super(SE3TransformerPooled, self).__init__()
        self.conv = ConvSE3(fiber_in[0], fiber_out[0])
        self.output_dim = output_dim

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

# Set up a minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a random graph
x = torch.randn(100, 5).to(device)  # Node features
edge_index = torch.randint(0, 100, (2, 100), dtype=torch.long).to(device)  # Edge indices

# Create a SE3-Transformer model with multiple input types
fiber_in = {0: 5, 1: 1}
fiber_out = {0: 10}
model = SE3TransformerPooled(fiber_in, fiber_out, output_dim=1).to(device)

# Trigger the bug
try:
    output = model(x, edge_index)
except RuntimeError as e:
    print(e)