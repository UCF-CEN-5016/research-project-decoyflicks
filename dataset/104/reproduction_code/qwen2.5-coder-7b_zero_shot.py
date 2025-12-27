import torch
import torch.nn as nn
from vector_quantize_pytorch.residual_vq import ResidualVQ


class VQModel(nn.Module):
    """
    Wrapper module around the ResidualVQ from vector_quantize_pytorch.
    Keeps the original behavior but provides clearer naming and structure.
    """

    def __init__(
        self,
        dims: int = 8,
        dim_z: int = 4,
        K: int = 16,
        L: int = 3,
        implicit_neural_codebook: bool = False,
    ) -> None:
        super().__init__()
        self._residual_vq = ResidualVQ(
            dims=dims, dim_z=dim_z, K=K, L=L, implicit_neural_codebook=implicit_neural_codebook
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._residual_vq(x)


if __name__ == "__main__":
    model = VQModel()
    sample_input = torch.randn(1, 8, 4)
    result = model(sample_input)
    print(result)