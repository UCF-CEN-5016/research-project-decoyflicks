import torch
from torch.nn import SE3TransformerPooled

class Fiber(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = torch.nn.Parameter(torch.tensor(dimensions))

    def forward(self, x):
        return x

data_config = {'NODE_FEATURE_DIM': 6}
extra_args = {}

se3_model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),
    fiber_out=Fiber({0: data_config['NODE_FEATURE_DIM'] * 2}),
    fiber_edge=Fiber({0: data_config['NODE_FEATURE_DIM']}),
    output_dim=1,
    tensor_cores=True,
    **vars(extra_args)
)

sample_input = torch.randn(8910, 3)
model_output = se3_model(sample_input)