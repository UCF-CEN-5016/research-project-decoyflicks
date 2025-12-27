import torch
from torch import nn
from torch_scatter import scatter
from collections import defaultdict

class Fiber(defaultdict):
    def __init__(self, d=None):
        super().__init__(int)
        if d is not None:
            for k, v in d.items():
                self[k] = v

class ConvSE3(nn.Module):
    def __init__(self, fiber_in, fiber_out):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.linear = nn.ModuleDict()
        for degree, dim_in in fiber_in.items():
            dim_out = fiber_out.get(degree, 0)
            if dim_out > 0:
                self.linear[str(degree)] = nn.Linear(dim_in, dim_out)

    def forward(self, features):
        out = {}
        for degree, x in features.items():
            if degree in self.linear:
                # x shape: [N, dim_in, *]
                # linear expects last dim dim_in, so permute
                x_perm = x.permute(0, 2, 1)
                y = self.linear[degree](x_perm)
                y = y.permute(0, 2, 1)
                out[degree] = y
            else:
                out[degree] = x
        return out

fiber_in = Fiber({0:5, 1:1})
fiber_out = Fiber({0:3, 1:0})

# batch size 8910, features per node (e.g. channels) dim_in, and "feature dimension" 1
x = {
    0: torch.randn(8910, 5, 1),
    1: torch.randn(8910, 1, 1)
}

conv = ConvSE3(fiber_in, fiber_out)
out = conv(x)