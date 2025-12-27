import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.pos_embedding = nn.Parameter(torch.randn(1, 197, 768))
    def forward(self, tokens, num_patches):
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1]
        return tokens

model = Model()
tokens = torch.randn(1, 196, 768)
num_patches = 196
model(tokens, num_patches)