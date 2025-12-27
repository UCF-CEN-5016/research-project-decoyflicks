import torch
from torch.nn import SE3TransformerPooled

datamodule = {'NODE_FEATURE_DIM': 6}
args = {}

class Fiber(torch.nn.Module):
    def __init__(self, dimensions):
        super(Fiber, self).__init__()
        self.dimensions = torch.nn.Parameter(torch.tensor(dimensions))

    def forward(self, x):
        return x

model = SE3TransformerPooled(
    fiber_in=Fiber({0: 5, 1: 1}),
    fiber_out=Fiber({0: datamodule['NODE_FEATURE_DIM'] * 2}),
    fiber_edge=Fiber({0: datamodule['NODE_FEATURE_DIM']}),
    output_dim=1,
    tensor_cores=True,
    **vars(args)
)

input_tensor = torch.randn(8910, 3)
output = model(input_tensor)