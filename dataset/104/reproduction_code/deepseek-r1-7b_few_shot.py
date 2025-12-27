import torch
from torch import nn

class ResidualVQ(nn.Module):
    def __init__(self, codebook_size, hidden_dim=64, embedding_dim=512):
        super().__init__()
        self.mlp_in = nn.Linear(codebook_size, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, codebook_size)

q_vq = ResidualVQ(10, 64, 512)  # Create an instance without considering the flag
print("MLP layers exist:", hasattr(q_vq, 'mlp_in') and hasattr(q_vq, 'mlp_out'))