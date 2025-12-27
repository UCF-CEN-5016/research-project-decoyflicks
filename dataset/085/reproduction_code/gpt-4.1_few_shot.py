import torch
import torch.nn as nn

class DummyEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # Positional embeddings: (1, num_patches+1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
    
    def forward(self, tokens):
        # Buggy line: missing closing bracket in slice
        # tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)
        # Correct line (for reference): tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)]
        
        # Reproduce the bug by executing the buggy line below in a try-except to show syntax error
        # But syntax error cannot be caught at runtime, so we simulate the bug by commenting it
        pass

num_patches = 16
embed_dim = 768
tokens = torch.randn(1, num_patches, embed_dim)
encoder = DummyEncoder(num_patches, embed_dim)

# The following simulates the buggy line (will cause SyntaxError if uncommented):
# tokens = tokens + encoder.pos_embedding[:, 1:(num_patches + 1)

# To demonstrate the intended operation without error:
tokens = tokens + encoder.pos_embedding[:, 1:(num_patches + 1)]

print("Tokens shape:", tokens.shape)