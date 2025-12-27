import torch
import torch.nn as nn
from typing import Tuple

class QKNormalizer(nn.Module):
    """Simple module that applies a per-kv-head normalization/scaling to queries.

    The normalization parameter has shape (num_kv_heads, 1, head_dim) so it can broadcast
    against query tensors shaped like (batch_or_one, num_heads, head_dim).
    """
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Parameter shaped to broadcast across queries of shape (1, num_heads, head_dim)
        self.qk_scale = nn.Parameter(torch.ones(num_kv_heads, 1, head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Multiply queries by the per-kv-head scale (relies on broadcasting)
        return q * self.qk_scale

if __name__ == "__main__":
    model = QKNormalizer(num_heads=8, num_kv_heads=4, head_dim=64)
    q = torch.randn(1, 8, 64)  # Query with 8 heads
    k = torch.randn(1, 4, 64)  # Key with 4 heads

    output = model(q, k, k)
    print(output.shape)