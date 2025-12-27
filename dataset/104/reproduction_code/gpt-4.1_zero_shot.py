import torch
from torch import nn

class ResidualVQ(nn.Module):
    def __init__(
        self,
        dim,
        num_residuals,
        codebook_size,
        codebook_dim,
        implicit_neural_codebook=False,
    ):
        super().__init__()
        self.implicit_neural_codebook = implicit_neural_codebook
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            for _ in range(num_residuals)
        ])

model = ResidualVQ(dim=16, num_residuals=2, codebook_size=10, codebook_dim=16, implicit_neural_codebook=False)
print(model.mlps)