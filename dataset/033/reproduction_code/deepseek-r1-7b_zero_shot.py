import torch
from torch.nn import functional as F
from se3transformer import SE3TransformerPooled, Fiber

class MinimalSE3TransformerPooled(SE3TransformerPooled):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, batched_graph):
        edge_index = batched_graph.edge_index
        edge_attr = batched_graph.ndata['attr']
        node_attr = batched_graph.ndata['attr']
        
        # Extract node features for each type
        node_feats0 = node_attr[:, :5]
        node_feats1 = node_attr[:, 5:6]
        
        batched_graph.ndata['node_attr'] = torch.cat([node_feats0, node_feats1], dim=1)
        
        edge_feats = batched_graph.ndata.get('edge_attr', None) if batched_graph.is_edge_case else None
        
        # Process edges
        edge feats are not the issue here; focus on node processing.
        
        return super().forward(batched_graph)

def main():
    model = MinimalSE3TransformerPooled(
        fiber_in=Fiber({0:5, 1:1}),
        fiber_out=Fiber({0:5, 1:5}), # Simplified for the example
        fiber_edge=Fiber({0:3}), # Assuming edge features dimension is 3
        output_dim=1,
    )
    
    # Simulate a batch with node attributes split into two types
    batched_graph = {
        'edge_index': torch.zeros(2, 8910).long(),
        'node_attr': torch.randn( (8910,6) ),  # Split into two parts: 5 and 1 for types 0 and 1
        'batch': torch.zeros(8910).long()
    }
    
    with torch.no_grad():
        out = model(batched_graph)
        print(out.shape)

if __name__ == '__main__':
    main()