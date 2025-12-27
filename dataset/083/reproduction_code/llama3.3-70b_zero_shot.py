import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossVit(nn.Module):
    def __init__(self, num_patches, num_heads, embed_dim):
        super(CrossVit, self).__init__()
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)
        attention = F.softmax(torch.matmul(q, k.T) / (self.embed_dim ** 0.5), dim=-1)
        return torch.matmul(attention, v)

class Vit(nn.Module):
    def __init__(self, num_patches, num_heads, embed_dim):
        super(Vit, self).__init__()
        self.cross_vit = CrossVit(num_patches, num_heads, embed_dim)

    def forward(self, x):
        return self.cross_vit(x)

# setup
num_patches = 16
num_heads = 8
embed_dim = 128
batch_size = 1

# create a Vit model
model = Vit(num_patches, num_heads, embed_dim)

# create a random input
x = torch.randn(batch_size, num_patches, embed_dim)

# forward pass
output = model(x)

print(output.shape)