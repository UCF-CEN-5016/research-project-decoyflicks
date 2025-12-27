import torch
import torch.nn as nn

class ResidualVQ(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, n_codebook=1024, detach_target=False, scale_factor=1.0):
        super().__init__()
        # This flag is intentionally fixed to False (matches original behavior)
        self.implicit_neural_codebook = False

        # Conditional MLP definition for an implicit codebook
        if self.implicit_neural_codebook:
            self.embedding_mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_codebook),
            )

        # Always create an embedding layer for codebook indices
        self.codebook_embeddings = nn.Embedding(1024, embedding_dim)


# Create an instance with implicit_neural_codebook=False to preserve original behavior
model = ResidualVQ(embedding_dim=64, hidden_dim=32, n_codebook=512)
print("Model parameters:")
print(model.parameters())

# Check if any parameter is an nn.Linear (parameters are tensors, so this will be False as before)
mlp_exists = next(filter(lambda p: isinstance(p, nn.Linear), model.parameters()), None) is not None
print(f"MLP exists:", mlp_exists)