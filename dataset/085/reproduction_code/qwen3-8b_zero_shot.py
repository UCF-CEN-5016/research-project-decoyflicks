import torch

class MAE:
    def __init__(self):
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, 5, 64))
    
    def forward(self, tokens, num_patches):
        tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)]
        return tokens

# Test case
mae = MAE()
tokens = torch.randn(1, 3, 64)
num_patches = 2
mae.forward(tokens, num_patches)