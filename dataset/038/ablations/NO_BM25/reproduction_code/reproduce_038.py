import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)
        self.cos_cached = torch.rand(4, 4, 8, 8)
        self.sin_cached = torch.rand(4, 4, 8, 8)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

batch_size = 4
channels = 4
x = torch.rand(batch_size, channels, 8, 8)
time_embedding = TimeEmbedding(channels * 4)
neg_half_x = torch.rand(batch_size, channels, 8, 8)

try:
    x_rope = (x * time_embedding.cos_cached[:x.shape[0], :, :, :]) + (neg_half_x * time_embedding.sin_cached[:x.shape[0], :, :, :])
except RuntimeError as e:
    print(e)
    print(f"x_rope shape: {x.shape}, cos_cached shape: {time_embedding.cos_cached.shape}, sin_cached shape: {time_embedding.sin_cached.shape}")