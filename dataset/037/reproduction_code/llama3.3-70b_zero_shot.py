import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ValuePE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        d_model = self.d_model
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device)

        inv_freq = 1 / (10000 ** (torch.arange(0, d_model, 2, device=x.device) / d_model))
        pe_even = torch.sin(position[:, None] * inv_freq[None, :])
        pe_odd = torch.cos(position[:, None] * inv_freq[None, :])

        pe = torch.cat((pe_even, pe_odd), dim=-1)
        pe = pe[:, None, :]

        x = x + pe
        x = x.permute(1, 0, 2)
        x = x.reshape(-1, x.size(-1))
        x = x.reshape(-1, 2, x.size(-1) // 2)
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(x.size(0), -1)

        x = x.permute(1, 0, 2)
        x = x.reshape(-1, x.size(-1))
        x = x.reshape(-1, 2, x.size(-1) // 2)
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(x.size(0), -1)

        return x

x = torch.randn(1, 10, 512)
model = ValuePE(512)
print(model(x).shape)