import torch
from torch_geometric.data import Data
from torch_geometric.nn import SE3TransformerPooled
from torch_geometric.utils import to_hetero
from typing import Dict

# Set up some dummy data
node_feats = {'0': torch.randn(8910, 1), '1': torch.randn(8910, 2)}
edge_index = torch.tensor([[0, 1], [1, 0]])
edge_attr = torch.randn(2, 6)
batch_size = 1

# Create a batched graph
batched_graph = Data(node_feats=node_feats, edge_index=edge_index, edge_attr=edge_attr)

# Set up the SE3Transformer model
fiber_in = Fiber({0: 5, 1: 1})
fiber_out = Fiber({0: 10, 1: 2})
fiber_edge = Fiber({0: 6})
model = SE3TransformerPooled(
    fiber_in=fiber_in,
    fiber_out=fiber_out,
    fiber_edge=fiber_edge,
    output_dim=1
)

try:
    output = model(batched_graph)
except RuntimeError as e:
    print(f"Error: {e}")