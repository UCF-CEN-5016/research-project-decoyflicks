import torch.nn as nn
from vector_quantize_pytorch.residual_vq import ResidualVQ

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vq = ResidualVQ(dims=8, dim_z=4, K=16, L=3, implicit_neural_codebook=False)

    def forward(self, x):
        return self.vq(x)

net = Net()