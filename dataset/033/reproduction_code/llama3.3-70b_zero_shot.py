import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

class Fiber:
    def __init__(self, dims):
        self.dims = dims

class SE3TransformerPooled(nn.Module):
    def __init__(self, fiber_in, fiber_out, fiber_edge, output_dim, **kwargs):
        super(SE3TransformerPooled, self).__init__()
        self.conv = ConvSE3(fiber_in, fiber_out, fiber_edge)

    def forward(self, batched_graph):
        node_feats = batched_graph.ndata['attr']
        output = self.conv(node_feats)
        return output

class ConvSE3(nn.Module):
    def __init__(self, fiber_in, fiber_out, fiber_edge):
        super(ConvSE3, self).__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.fiber_edge = fiber_edge
        self.lin = nn.Linear(fiber_in[0] + fiber_edge[0], fiber_out[0])

    def forward(self, node_feats):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.randn(2, self.fiber_edge[0])
        node_attr = torch.randn(2, self.fiber_in[0])
        output = torch.cat([node_attr, edge_attr], dim=1)
        output = self.lin(output)
        return output

class QM9DataModule:
    def __init__(self):
        self.NODE_FEATURE_DIM = 6
        self.EDGE_FEATURE_DIM = 1

    def get_data(self):
        node_attr = torch.randn(8910, self.NODE_FEATURE_DIM)
        edge_attr = torch.randn(2, self.EDGE_FEATURE_DIM)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        batched_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        return batched_graph

args = type('args', (object,), {'amp': False, 'num_degrees': 1, 'num_channels': 1})
model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}), 
    fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
    fiber_edge=Fiber({0: 1}),
    output_dim=1
)

datamodule = QM9DataModule()
batched_graph = datamodule.get_data()
batched_graph.ndata['attr'] = batched_graph.x

node_feats = {'0': batched_graph.ndata['attr'][:, :5, None],
               '1': batched_graph.ndata['attr'][:, 5:6, None]}
batched_graph.ndata['attr'] = torch.cat([node_feats['0'], node_feats['1']], dim=1)

model(batched_graph)