import torch
import torch.nn as nn
from typing import Optional
from x_transformers import XTransformers, KVHeads


class Model(XTransformers):
    """
    Lightweight wrapper around XTransformers preserving original behavior.
    Exposes clearer internal names while remaining backwards-compatible
    with the original attribute names (kv_heads, heads, qk_norm_k_scale).
    """

    def __init__(
        self,
        kv_heads_count: int = 2,
        num_heads: int = 1,
        qk_norm_scale_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Initialize underlying storage for renamed attributes
        self._kv_heads_count = kv_heads_count
        self._num_heads = num_heads
        if qk_norm_scale_init is None:
            qk_norm_scale_init = torch.randn(1)
        self._qk_norm_scale = nn.Parameter(qk_norm_scale_init)

    # Backwards-compatible properties matching original attribute names

    @property
    def kv_heads(self) -> int:
        return self._kv_heads_count

    @kv_heads.setter
    def kv_heads(self, value: int) -> None:
        self._kv_heads_count = value

    @property
    def heads(self) -> int:
        return self._num_heads

    @heads.setter
    def heads(self, value: int) -> None:
        self._num_heads = value

    @property
    def qk_norm_k_scale(self) -> nn.Parameter:
        return self._qk_norm_scale

    @qk_norm_k_scale.setter
    def qk_norm_k_scale(self, value: nn.Parameter) -> None:
        self._qk_norm_scale = value


model = Model()