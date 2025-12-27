import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.utils import Fiber

batch_size = 10
num_nodes = 100
edge_feat_dim = 4

model = SE3Transformer(
    fiber_in=Fiber({0: 5, 1: 1}),
    fiber_out=Fiber({0: 1}),
    fiber_edge=Fiber({0: edge_feat_dim}),
    depth=1,
    heads=1
)

graph = {
    'edges': torch.randint(0, num_nodes, (2, batch_size * 20)),
    'nodes': {
        '0': torch.randn(batch_size * num_nodes, 5, 1),
        '1': torch.randn(batch_size * num_nodes, 1, 1)
    },
    'edges_features': torch.randn(batch_size * 20, edge_feat_dim, 1)
}

output = model(graph)