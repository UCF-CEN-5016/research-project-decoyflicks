import torch
from torch import nn

# Minimal mock of ResidualVQ focusing on MLP initialization logic
class ResidualVQ(nn.Module):
    def __init__(self, dim, num_residuals, implicit_neural_codebook=False):
        super().__init__()
        self.dim = dim
        self.num_residuals = num_residuals
        self.implicit_neural_codebook = implicit_neural_codebook
        
        # Bug: MLPs are always initialized regardless of implicit_neural_codebook flag
        # Original code snippet from the bug report:
        # self.mlps = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(dim, dim * 2),
        #         nn.ReLU(),
        #         nn.Linear(dim * 2, dim)
        #     )
        #     for _ in range(num_residuals)
        # ])
        
        # The bug is that this is done even if implicit_neural_codebook=False
        # So let's reproduce that exactly:
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim)
            )
            for _ in range(num_residuals)
        ])

# Test to confirm MLPs exist even if implicit_neural_codebook=False
vq = ResidualVQ(dim=8, num_residuals=2, implicit_neural_codebook=False)
print(f"implicit_neural_codebook={vq.implicit_neural_codebook}")
print(f"Number of MLPs initialized: {len(vq.mlps)}")  # Expected: 0 if correct, but bug causes 2

# Check the presence of MLP modules explicitly
for i, mlp in enumerate(vq.mlps):
    print(f"MLP {i} structure: {mlp}")