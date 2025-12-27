import torch
from vector_quantize_pytorch.residual_sim_vq import ResidualSimVQ

class MyModel(ResidualSimVQ):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
        pass

my_model = MyModel()
x = torch.randn(2, 1024, 17)
my_model(x)