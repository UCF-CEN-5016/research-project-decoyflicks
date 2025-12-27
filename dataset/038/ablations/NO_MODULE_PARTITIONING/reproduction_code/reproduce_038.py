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

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = TimeEmbedding(32)
        self.cos_cached = torch.rand(4, 4, 32, 32)
        self.sin_cached = torch.rand(4, 4, 32, 32)

    def forward(self, x):
        x_rope = torch.zeros_like(x)
        neg_half_x = torch.rand_like(x)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

batch_size = 4
channels = 4
x = torch.rand(batch_size, channels, 32, 32)

model = TestModel()
try:
    model(x)
except RuntimeError as e:
    print(e)
    print(f"x_rope shape: {x.shape}, cos_cached shape: {model.cos_cached.shape}, sin_cached shape: {model.sin_cached.shape}")