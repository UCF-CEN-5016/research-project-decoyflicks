import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.token_embedding = nn.Embedding(10, 10)
        self.position_embedding = nn.Parameter(torch.randn(1, 11, 10))

    def forward(self):
        token_weights = self.token_embedding.weight
        return token_weights + self.position_embedding[:, 1:(2 + 1)]

mae = MAE()