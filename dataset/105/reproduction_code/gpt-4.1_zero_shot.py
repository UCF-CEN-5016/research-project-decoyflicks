import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyEmbed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embed = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def replace(self, ind, mask, sampled):
        # This mimics the problematic line:
        # self.embed.data[ind][mask] = sampled
        self.embed.data[ind][mask] = sampled

    def forward(self, x):
        # Random indices for replacement
        ind = torch.randint(0, self.num_embeddings, (x.size(0)*2,))
        # Create a boolean mask with one True missing compared to sampled
        mask = torch.zeros(ind.size(0), dtype=torch.bool)
        # Set almost all True except one to cause mismatch
        mask[:len(ind)//2] = True
        # sampled has one less row than mask True count -> causes mismatch
        sampled = torch.randn(mask[:len(ind)//2].sum().item() - 1, self.embedding_dim)
        self.replace(ind, mask, sampled)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = DummyEmbed(16384, 512)

    def forward(self, x):
        return self.dummy(x)

model = Model()
x = torch.randn(8, 512)

for _ in range(100):
    try:
        model(x)
    except RuntimeError as e:
        print(e)
        break