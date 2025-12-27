import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.cos_cached = torch.cos(torch.arange(max_seq_len) * (-1 / 10000 ** (2 * torch.arange(0, dim, 2) / dim)))
        self.sin_cached = torch.sin(torch.arange(max_seq_len) * (-1 / 10000 ** (2 * torch.arange(0, dim, 2) / dim)))

    def forward(self, x):
        x_rope = x
        neg_half_x = -x / 2
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

if __name__ == "__main__":
    max_seq_len = 100
    dim = 10
    seq_len = 4
    x = torch.randn(3, seq_len, dim)
    rope = RotaryPositionalEmbeddings(dim, max_seq_len)
    print(rope(x))