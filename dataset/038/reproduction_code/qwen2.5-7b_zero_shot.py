import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, seq_len=10, num_layers=2, embedding_dim=4):
        super().__init__()
        self.cos_cached = nn.Parameter(torch.randn(seq_len, num_layers, embedding_dim))
        self.sin_cached = nn.Parameter(torch.randn(seq_len, num_layers, embedding_dim))

    def forward(self, x):
        seq_len = x.shape[0]
        cos_slice = self.cos_cached[:seq_len]
        sin_slice = self.sin_cached[:seq_len]
        x_rope = x * cos_slice + (-x) * sin_slice
        return x_rope

model = RotaryEmbedding()
x = torch.randn(5, 2, 3)
output = model(x)
print(output)