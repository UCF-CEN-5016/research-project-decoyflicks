import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = torch.randn(1, 10, 32)

    def forward(self, tokens, num_patches):
        tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)
        return tokens

encoder = Encoder()
tokens = torch.randn(1, 5, 32)
num_patches = 5
output = encoder(tokens, num_patches)