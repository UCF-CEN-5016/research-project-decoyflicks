import torch
from torch_geometric.data import Data
from se3_transformer import SE3TransformerPooled, Fiber  # Import Fiber from the correct module

# Define dummy data
batch_size = 8910
input_dim = 6
dummy_node_features = torch.randn(batch_size, input_dim)
dummy_edge_features = torch.randn(batch_size, 3)

# Create a dummy batched graph
edge_index = torch.tensor([[i for i in range(batch_size)] * 2], dtype=torch.long).t().contiguous()
batched_graph = Data(x=dummy_node_features, edge_index=edge_index, edge_attr=dummy_edge_features)

# Initialize SE3TransformerPooled model
fiber_in = Fiber({0: 5, 1: 1})
fiber_out = Fiber({0: 1})
fiber_edge = Fiber({0: 3})
model = SE3TransformerPooled(
    fiber_in=fiber_in,
    fiber_out=fiber_out,
    fiber_edge=fiber_edge,
    output_dim=1
)

# Forward pass to trigger the error
output = model(batched_graph)