import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.utils import Fiber

# Minimal reproduction of the bug
def reproduce_bug():
    # Create model with split input features (type 0:5 dim, type 1:1 dim)
    model = SE3Transformer(
        fiber_in=Fiber({0: 5, 1: 1}),
        fiber_out=Fiber({0: 4}),
        fiber_edge=Fiber({0: 4}),
        depth=1,
        num_heads=4,
        dim_head=16
    )

    # Create dummy graph data
    num_nodes = 100
    edges = torch.randint(0, num_nodes, (2, 200))  # random edges
    node_feats = {
        '0': torch.randn(num_nodes, 5, 1),  # type 0 features
        '1': torch.randn(num_nodes, 1, 1)   # type 1 features
    }
    edge_feats = {'0': torch.randn(200, 4, 1)}

    # This will trigger the dimension mismatch error
    try:
        output = model(node_feats, edges, edge_feats)
    except RuntimeError as e:
        print("Error reproduced successfully!")
        print(f"Error message: {str(e)}")

if __name__ == '__main__':
    reproduce_bug()