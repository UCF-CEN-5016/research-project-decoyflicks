import torch
from vector_quantize_pytorch.residual_sim_vq import ResidualSimVQ


class MyModel(ResidualSimVQ):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Determine whether to apply quantize dropout according to training state
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
        pass


if __name__ == "__main__":
    model = MyModel()
    x = torch.randn(2, 1024, 17)
    model(x)