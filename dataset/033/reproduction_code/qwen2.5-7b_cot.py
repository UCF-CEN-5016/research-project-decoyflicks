import torch
from torch_geometric.data import Data
from SE3Transformer import SE3TransformerPooled, Fiber

# Create mock data with split node features
num_nodes = 10
num_edges = 20

# Create mock node features
node_features_type_0 = torch.rand(num_nodes, 5)
node_features_type_1 = torch.rand(num_nodes, 1)
node_features = torch.cat([node_features_type_0, node_features_type_1], dim=1)

# Create mock edge indices
edge_index = torch.randint(0, num_nodes, (2, num_edges))

# Create mock graph
mock_graph = Data(x=node_features, edge_index=edge_index)

# Initialize model with modified fiber_in
model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),
    fiber_out=Fiber({0: 2 * 16}),
    fiber_edge=Fiber({0: 4}),
    output_dim=1,
    tensor_cores=False
)

# Forward pass to trigger the error
with torch.no_grad():
    output = model(mock_graph)
    print(output.shape)

# Note: The following line causes an error since 'input' and 'weight' are not defined
# output = torch.matmul(input, weight)  # Requires matching channel dimensions