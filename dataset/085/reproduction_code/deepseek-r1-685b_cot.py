import torch
import torch.nn as nn

class BuggyMAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate a positional embedding with shape (1, 100, 768)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, 768))
        
    def forward(self, tokens, num_patches):
        # The buggy line - trying to access pos_embedding beyond its dimension
        tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)]
        return tokens

# Reproduction case
model = BuggyMAE()
dummy_tokens = torch.randn(1, 50, 768)  # Batch of 50 tokens
num_patches = 99  # This will trigger the bug since 1:100 is out of bounds

# This will raise IndexError since pos_embedding only has 100 elements
# along dim 1, but we're trying to access up to index 100 (1:100)
output = model(dummy_tokens, num_patches)