import torch
import torch.nn as nn

class ResidualVQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, implicit_neural_codebook=False):
        super(ResidualVQ, self).__init__()
        self.implicit_neural_codebook = implicit_neural_codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.mlp = nn.Linear(embedding_dim, embedding_dim)

model = ResidualVQ(num_embeddings=512, embedding_dim=64, implicit_neural_codebook=False)
print(list(model.named_parameters()))