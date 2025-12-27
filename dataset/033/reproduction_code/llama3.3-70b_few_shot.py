import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class SE3Transformer(nn.Module):
    def __init__(self, fiber_in, fiber_out, fiber_edge, output_dim):
        super(SE3Transformer, self).__init__()
        self.conv = pyg_nn.SE3TransformerConv(fiber_in, fiber_out, fiber_edge)

    def forward(self, data):
        out = self.conv(data.x, data.edge_index)
        return out

# Define node features with multiple input types
node_feats = {
    '0': torch.randn(8910, 5),  # First input type
    '1': torch.randn(8910, 1)  # Second input type
}

# Define edge features
edge_feats = torch.randn(8910, 1)

# Create a dummy graph
x = torch.cat((node_feats['0'], node_feats['1']), dim=1)  # Concatenate node features
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).T  # Dummy edge index

# Create a Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_feats)

# Initialize the SE3Transformer model
model = SE3Transformer(
    fiber_in={'0': 5, '1': 1},
    fiber_out={'0': 10},  # Output fiber
    fiber_edge={'0': 1},  # Edge fiber
    output_dim=1
)

# Forward pass that causes the error
try:
    out = model(data)
    print(out.shape)
except RuntimeError as e:
    print(e)