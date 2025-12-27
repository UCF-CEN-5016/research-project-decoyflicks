import torch
from e3nn import o3
from e3nn.math import softplus
from e3nn.nn import feature_transformer

# Define Fiber class (simplified for reproduction)
class Fiber:
    def __init__(self, specs):
        self.specs = specs
        
    def __getitem__(self, key):
        return self.specs[key]

# Mock SE3Transformer implementation (simplified)
class SE3TransformerPooled:
    def __init__(self, fiber_in, fiber_out, fiber_edge, output_dim, **kwargs):
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.fiber_edge = fiber_edge
        self.output_dim = output_dim
        
        # Simulate internal operations that expect specific input dimensions
        self.expected_input_channels = 6  # Original expected input size
        
    def forward(self, x):
        # Simulate self-attention operation that fails with multiple input types
        if len(self.fiber_in.specs) > 1:
            raise RuntimeError(f"Expected size for first two dimensions of batch2 tensor to be: [{self.expected_input_channels}, 1] but got: [{sum(self.fiber_in.specs.values())}, 1].")
        
        # Simulate successful forward pass
        return torch.randn(x.shape[0], self.output_dim)

# Simulate data loading with modified node features
def load_data():
    # Simulate batched graph with 8910 nodes
    batched_graph = type('Graph', (), {
        'ndata': {
            'attr': torch.randn(8910, 6)  # Original 6 features
        }
    })
    
    # Split features into two types (5 and 1)
    node_feats = {
        '0': batched_graph.ndata['attr'][:, :5, None],
        '1': batched_graph.ndata['attr'][:, 5:6, None]
    }
    
    return node_feats

# Main execution
if __name__ == "__main__":
    # Load data with modified node features
    node_feats = load_data()
    
    # Create model with fiber_in expecting single type (original setup)
    model = SE3TransformerPooled(
        fiber_in=Fiber({0: 5, 1: 1}),  # Modified input with two types
        fiber_out=Fiber({0: 3}),       # Output configuration
        fiber_edge=Fiber({0: 3}),      # Edge features
        output_dim=1
    )
    
    # Simulate forward pass that triggers the error
    try:
        output = model(node_feats)
        print("Success:", output.shape)
    except RuntimeError as e:
        print("Error occurred:", e)