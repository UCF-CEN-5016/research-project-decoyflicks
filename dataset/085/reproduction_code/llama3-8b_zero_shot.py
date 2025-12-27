import torch.nn as nn

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(10, 10)
        self.pos_embedding = nn.Parameter(torch.randn(1, 11, 10))

    def forward(self):
        tokens = self.encoder.weight
        return tokens + self.pos_embedding[:, 1:(2 + 1)]

mae = MAE()