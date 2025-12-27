import torch
import torch_geometric
from torch_geometric.data import Data
from SE3Transformer import SE3TransformerPooled, Fiber  # Assuming the SE3Transformer is available

# 1. Create mock data with split node features
num_nodes = 10
num_edges = 20

# Create mock node features: split into two types (0: 5 features, 1: 1 feature)
node_features = torch.cat([
    torch.rand(num_nodes, 5),  # Type 0: 5 features
    torch.rand(num_nodes, 1)   # Type 1: 1 feature
], dim=1)  # Shape: [num_nodes, 6]

# Create mock edge indices
edge_index = torch.randint(0, num_nodes, (2, num_edges))

# Create mock graph
mock_graph = Data(
    x=node_features,  # Shape: [num_nodes, 6]
    edge_index=edge_index
)

# 2. Initialize model with modified fiber_in (expecting 2 node types)
model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),  # Split node types
    fiber_out=Fiber({0: 2 * 16}),  # Example output
    fiber_edge=Fiber({0: 4}),      # Example edge features
    output_dim=1,
    tensor_cores=False
)

# 3. Forward pass to trigger the error
with torch.no_grad():
    output = model(mock_graph)
    print(output.shape)

output = torch.matmul(input, weight)  # Requires matching channel dimensions