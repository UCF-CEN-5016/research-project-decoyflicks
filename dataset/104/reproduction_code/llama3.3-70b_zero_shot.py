import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualVQ(nn.Module):
    def __init__(self, dim, num_vectors, beta, implicit_neural_codebook=False):
        super().__init__()
        self.implicit_neural_codebook = implicit_neural_codebook
        self.dim = dim
        self.num_vectors = num_vectors
        self.beta = beta
        self.embedding = nn.Embedding(self.num_vectors, self.dim)
        if self.implicit_neural_codebook:
            self.mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim)
            )

    def forward(self, x):
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        dvec = self.embedding.weight
        dvec = dvec.t().unsqueeze(0).repeat(x_flat.shape[0], 1, 1)
        dist = (dvec ** 2).sum(d=2) - 2 * torch.einsum("bd,dn->bn", x_flat, dvec) + (x_flat ** 2).sum(d=1, keepdim=True)
        min_dist, min_encoding_indices = torch.min(dist, dim=1)
        z = self.embedding(min_encoding_indices).view(x.shape)
        if self.implicit_neural_codebook:
            z = z + self.mlp(x - z)
        else:
            z = z + nn.Linear(self.dim, self.dim)(x - z) # MLP is initialized here
        loss = self.beta * min_dist.mean()
        return z, loss

model = ResidualVQ(dim=128, num_vectors=128, beta=0.25, implicit_neural_codebook=False)
x = torch.randn(1, 16, 128)
z, loss = model(x)