import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> None:
        super().__init__()
        # store caches as buffers so they move with the module (cpu/cuda) but are not parameters
        self.register_buffer("_cos_cache", cos_cache)
        self.register_buffer("_sin_cache", sin_cache)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_len = tensor.shape[0]
        cos = self._cos_cache[:batch_len]
        sin = self._sin_cache[:batch_len]
        # keep the original arithmetic and broadcasting behavior intact
        return tensor * cos + (tensor[:, 1:] * -0.5) * sin


cos_cache = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
sin_cache = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
rope = RotaryPositionalEmbedding(cos_cache, sin_cache)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x_rope = rope(x)