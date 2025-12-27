import torch
from typing import Tuple

class RotaryEmbedding:
    """
    A minimal rotary embedding container that holds cached cosine and sine tensors
    and applies a rotary transform to an input tensor. The core logic is preserved.
    """
    def __init__(self, batch_size: int = 2, seq_len: int = 5, hidden_dim: int = 4):
        self._cos_cache = torch.randn(batch_size, seq_len, hidden_dim)
        self._sin_cache = torch.randn(batch_size, seq_len, hidden_dim)

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotary operation to input tensor x using cached cos/sin tensors.
        The operation intentionally mirrors the original logic (including shape handling).
        """
        batch, seq_len, _ = x.shape
        zeros = torch.zeros_like(x)  # placeholder for the negative-half rotated part
        cos_slice = self._cos_cache[:batch]
        sin_slice = self._sin_cache[:batch]
        rotated = (x * cos_slice) + (zeros * sin_slice)
        return rotated

    def __repr__(self) -> str:
        return f"RotaryEmbedding(cos_cache_shape={tuple(self._cos_cache.shape)}, sin_cache_shape={tuple(self._sin_cache.shape)})"


# Test case
rotary = RotaryEmbedding()
x = torch.randn(2, 5, 3)  # shape (batch, seq, 3)
try:
    result = rotary.apply_rotary(x)
    print(result.shape)
except RuntimeError as e:
    print(f"Error: {e}")