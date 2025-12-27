import torch
from torch import nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('cos_cached', torch.cos(position * div_term))
        self.register_buffer('sin_cached', torch.sin(position * div_term))

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, features]
        if x.shape[-1] != self.cos_cached.size(-1):
            # Select only the first d features
            cos = self.cos_cached[:, :x.shape[0], :, :x.shape[-1]]
            sin = self.sin_cached[:, :x.shape[0], :, :x.shape[-1]]
        else:
            cos = self.cos_cached[:x.shape[0], :, :, :]
            sin = self.sin_cached[:x.shape[0], :, :, :]
        
        x_rope = (x * cos) + (-0.5 * x * sin)
        return x_rope

import torch
import math

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('cos_cached', torch.cos(position * div_term))
        self.register_buffer('sin_cached', torch.sin(position * div_term))

    def forward(self, x):
        seq_len = x.shape[1]
        features = x.shape[-1]
        
        # Slice only the needed positions and features
        cos = self.cos_cached[:, :seq_len, :, :features]
        sin = self.sin_cached[:, :seq_len, :, :features]
        
        # Apply rotary positional encoding
        x_rotated = (x * cos) - (x.permute(0, 2, 1, 3) * sin).permute(0, 2, 1, 3)
        return x_rotated

import torch
from torch import nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('cos_cached', torch.cos(position * div_term))
        self.register_buffer('sin_cached', torch.sin(position * div_term))

    def forward(self, x):
        seq_len = x.shape[1]
        features = x.shape[-1]
        
        # Slice only the needed positions and features
        cos = self.cos_cached[:, :seq_len, :, :features]
        sin = self.sin_cached[:, :seq_len, :, :features]
        
        return (x * cos) + (-0.5 * x * sin)

# Example usage:
model = RotaryPositionalEncoding(32)
input = torch.randn(16, 8, 10)  # batch_size=16, seq_len=8, features=10
output = model(input)
print(output.shape)  # Should output: torch.Size([16, 8, 10])