import torch
from se3transformer_pytorch import SE3TransformerPooled, Fiber

# Simulated batched graph node features (8910 nodes, 6 features)
# Splitting features into two fibers: one with 5 dims, another with 1 dim, each with channel dim=1
node_feats = {
    '0': torch.randn(8910, 5, 1),  # Fiber 0: 5 features, 1 channel
    '1': torch.randn(8910, 1, 1)   # Fiber 1: 1 feature, 1 channel
}

# Construct input fiber matching node_feats
fiber_in = Fiber({0: 5, 1: 1})

# Dummy edge fiber (e.g., single scalar edge features)
fiber_edge = Fiber({0: 4})  # arbitrary edge feature dim

# Model parameters
num_degrees = 1
num_channels = 1

# Initialize SE3TransformerPooled with multiple input fibers
model = SE3TransformerPooled(
    fiber_in=fiber_in,
    fiber_out=Fiber({0: num_degrees * num_channels}),
    fiber_edge=fiber_edge,
    output_dim=1,
    tensor_cores=False,
)

# Forward pass with inputs as a dict simulating batched_graph.ndata
# (In real use, the graph and edge features would be provided, here we mock edge_feats)
edge_feats = torch.randn(8910, 4, 1)  # dummy edge features

try:
    # This will raise RuntimeError due to size mismatch inside ConvSE3
    out = model(node_feats, edge_feats)
except RuntimeError as e:
    print("RuntimeError:", e)