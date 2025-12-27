import torch
import torch.nn as nn

class ResidualVQ(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, n_codebook=1024, detach_target=False, scale_factor=1.0):
        super().__init__()
        
        self.implicit_neural_codebook = False
        
        # MLP for the codebook (if implicit)
        if self.implicit_neural_codebook:
            self.embedding_MLP = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_codebook)
            )
        
        else:
            pass
        
        # Always create embedding layer
        self.x_embed = nn.Embedding(1024, embedding_dim)

# Create an instance with implicit_neural_codebook=False to trigger the bug
model = ResidualVQ(embedding_dim=64, hidden_dim=32, n_codebook=512)
print("Model parameters:")
print(model.parameters())

# Check if MLP layers exist (should not when flag is False)
mlp_exists = next(filter(lambda p: isinstance(p, nn.Linear), model.parameters()), None) is not None
print(f"MLP exists:", mlp_exists)