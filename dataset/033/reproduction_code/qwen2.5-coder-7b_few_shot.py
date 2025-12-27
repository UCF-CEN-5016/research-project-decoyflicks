import torch
from e3nn import o3
from e3nn.math import softplus
from e3nn.nn import feature_transformer

class Fiber:
    def __init__(self, specs):
        self.specs = specs

    def __getitem__(self, key):
        return self.specs[key]

    def total_channels(self):
        return sum(self.specs.values())

class SE3TransformerPooled:
    def __init__(self, fiber_in, fiber_out, fiber_edge, output_dim, **kwargs):
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.fiber_edge = fiber_edge
        self.output_dim = output_dim
        self.expected_input_channels = 6

    def forward(self, x):
        if self.fiber_in.total_channels() != self.expected_input_channels:
            raise RuntimeError(
                f"Expected size for first two dimensions of batch2 tensor to be: [{self.expected_input_channels}, 1] but got: [{self.fiber_in.total_channels()}, 1]."
            )

        return torch.randn(x.shape[0], self.output_dim)

def build_node_features():
    Graph = type('Graph', (), {'ndata': {'attr': torch.randn(8910, 6)}})
    graph_stub = Graph()

    node_feats = {
        '0': graph_stub.ndata['attr'][:, :5, None],
        '1': graph_stub.ndata['attr'][:, 5:6, None]
    }

    return node_feats

if __name__ == "__main__":
    node_feats = build_node_features()

    model = SE3TransformerPooled(
        fiber_in=Fiber({0: 5, 1: 1}),
        fiber_out=Fiber({0: 3}),
        fiber_edge=Fiber({0: 3}),
        output_dim=1
    )

    try:
        output = model(node_feats)
        print("Success:", output.shape)
    except RuntimeError as e:
        print("Error occurred:", e)