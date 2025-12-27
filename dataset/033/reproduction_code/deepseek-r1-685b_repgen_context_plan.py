import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.fiber import Fiber

# Define the SE3Transformer model with modified input configuration
model = SE3Transformer(
    fiber_in=Fiber({0: 5, 1: 1}),  # Two input types
    fiber_out=Fiber({0: 1}),
    fiber_edge=Fiber({0: 4}),  # Edge features
    num_degrees=2,
    num_channels=4,
    depth=1
)

# Generate sample data with multiple input types
nodes = {
    '0': torch.randn(10, 5, 1),  # Type 0 features
    '1': torch.randn(10, 1, 1)   # Type 1 features
}
edges = {
    '0': torch.randn(10, 4, 1)    # Edge features
}
edges_idx = torch.randint(0, 10, (2, 20))  # Random edges

# Handle potential dimension mismatch errors
try:
    # Forward pass through the model
    output = model(nodes, edges, edges_idx)
    print("Success!")
except RuntimeError as e:
    # Print error message and potential fix
    print(f"Error: {e}")
    print("The error occurs because the attention calculation expects consistent dimensions")
    print("when processing multiple input types. The edge features may need adjustment.")