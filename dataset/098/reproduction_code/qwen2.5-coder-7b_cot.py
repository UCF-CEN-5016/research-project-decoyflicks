import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    Produces embeddings of size `embedding_dim` for scalar inputs `x`.
    """
    def __init__(self, embedding_dim: int, theta: float = 10000.0):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.theta = float(theta)

    def _compute_frequencies(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        log_theta = math.log(self.theta) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -log_theta)
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1)
        device = x.device
        dtype = x.dtype

        freqs = self._compute_frequencies(device, dtype)  # (half_dim,)

        # Outer product: (N, 1) * (1, half_dim) -> (N, half_dim)
        emb = x_flat[:, None] * freqs[None, :]

        # Concatenate sin and cos
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        # If embedding_dim is odd, pad one zero column to match requested size
        if emb.size(-1) < self.embedding_dim:
            pad_size = self.embedding_dim - emb.size(-1)
            emb = torch.cat([emb, torch.zeros(emb.size(0), pad_size, device=device, dtype=dtype)], dim=-1)

        # Restore original leading shape, append embedding dim
        return emb.view(*original_shape, self.embedding_dim)


# Example usage
if __name__ == "__main__":
    emb_module = SinusoidalPosEmb(embedding_dim=4, theta=10000.0)
    x = torch.ones(10)
    out = emb_module(x)
    print(out, out.shape)