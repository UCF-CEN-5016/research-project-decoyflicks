import torch

class MAE(torch.nn.Module):
    def __init__(self, num_patches=5, embedding_dim=64):
        super(MAE, self).__init__()
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches, embedding_dim))
    
    def forward(self, tokens, num_patches):
        tokens = tokens + self.pos_embedding[:, :num_patches]
        return tokens

# Test case
mae = MAE()
tokens = torch.randn(1, 3, 64)
num_patches = 2
mae.forward(tokens, num_patches)