import torch
from torch import nn

def _exists(x):
    return x is not None

class RotaryEmbedding(nn.Module):
    """
    Refactored rotary embedding module.
    Core behavior preserved:
    - Keeps an optional cache buffer.
    - Uses a freqs buffer for frequency values.
    - forward is decorated with torch.no_grad().
    """
    def __init__(self, freqs: torch.Tensor = None):
        super().__init__()
        # cache is an optional buffer (can be None)
        self.register_buffer('cache', None)
        # freqs may be provided at construction; otherwise left uninitialized (None)
        if freqs is not None:
            self.register_buffer('freqs', freqs)
        else:
            self.freqs = None

    @torch.no_grad()
    def forward(self, t: torch.Tensor, seq_len: int = None, offset: int = 0) -> torch.Tensor:
        """
        Compute rotary frequencies for inputs `t` using stored self.freqs.

        Behavior:
        - If a cache exists and the requested slice is contained, return that slice.
        - Compute freqs_out = einsum('... , f -> ...f', t, self.freqs) and then
          repeat-interleave the last dimension by 2 (equivalent to einops repeat(... (n r), r=2)).
        - If a cache exists and the requested slice fits, write the computed values into the cache.
        """
        if self.cache is not None and _exists(seq_len) and (offset + seq_len) <= self.cache.shape[0]:
            return self.cache[offset:(offset + seq_len)]

        freqs_tensor = self.freqs
        if freqs_tensor is None:
            raise RuntimeError("RotaryEmbedding.freqs is not initialized")

        # Compute outer-product-like broadcasting: result shape ... x f
        freqs_out = torch.einsum('... , f -> ...f', t.type(freqs_tensor.dtype), freqs_tensor)
        # Repeat each frequency value twice along the last dimension (equivalent to einops repeat r=2)
        freqs_out = freqs_out.repeat_interleave(2, dim=-1)

        if self.cache is not None and _exists(seq_len) and (offset + seq_len) <= self.cache.shape[0]:
            self.cache[offset:(offset + seq_len)] = freqs_out

        return freqs_out

    @torch.no_grad()
    def init_cache_from_template(self, seq_length: int, template: torch.Tensor):
        """
        Initialize the cache buffer with zeros using the provided template to infer shape, dtype, and device.
        Cache shape will be (seq_length, *template.shape[1:]).
        """
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        shape = (seq_length,) + template.shape[1:]
        self.cache = torch.zeros(shape, dtype=template.dtype, device=template.device)

if __name__ == "__main__":
    # Example usage
    # Initialize freqs and model
    freqs = torch.linspace(1.0, 4.0, steps=8)  # example frequency vector
    model = RotaryEmbedding(freqs=freqs)

    # Example input tensor `t`. Shape should be compatible with einsum's '...' spec.
    some_tensor = torch.randn(2, 3)  # '...' can be (2,3) here

    # Forward pass (note: forward is no_grad -> output requires_grad is False)
    out = model(some_tensor)

    # Attempting to call backward on a tensor that does not require grad will raise.
    try:
        out.sum().backward()
    except RuntimeError as e:
        print(f"RuntimeError: {e}")