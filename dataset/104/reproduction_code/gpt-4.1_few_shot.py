import torch
from torch import nn

# Minimal mock of ResidualVQ with relevant behavior
class ResidualVQ(nn.Module):
    def __init__(self, dim, n_embed, n_residual, implicit_neural_codebook=False):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.n_residual = n_residual
        self.implicit_neural_codebook = implicit_neural_codebook
        
        # Bug: MLPs are initialized regardless of implicit_neural_codebook flag
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            for _ in range(n_residual)
        ])
        
        # If implicit_neural_codebook=False, MLPs should NOT be initialized,
        # but here they always are.
        
# Instantiate with implicit_neural_codebook=False
vq = ResidualVQ(dim=16, n_embed=8, n_residual=2, implicit_neural_codebook=False)

# Check if MLPs exist despite flag being False
print("Number of MLPs initialized:", len(vq.mlps))  # Expected: 0, Actual: 2

# This shows MLPs are always initialized, which is a bug