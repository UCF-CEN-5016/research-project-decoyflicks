import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.pos_embedding = nn.Parameter(torch.randn(10, 20))  # Position embedding

    def forward(self, tokens, num_patches):
        # Incorrect slicing of position embedding
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]  # Off-by-one error
        return tokens

# Sample input
tokens = torch.randn(1, 10, 20)  # Input tokens
num_patches = 10  # Number of patches

# Create MAE model
mae = MAE()

# Forward pass that produces error
try:
    output = mae(tokens, num_patches)
    print("Output shape:", output.shape)
except IndexError as e:
    print("Error:", e)