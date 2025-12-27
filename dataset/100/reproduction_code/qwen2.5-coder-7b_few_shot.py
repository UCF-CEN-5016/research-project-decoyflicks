import torch


class RefactoredCacheLayer(torch.nn.Module):
    """A simple module that caches the computed output (input * 2)."""
    def __init__(self) -> None:
        super().__init__()
        self._cached = None

    def _compute_and_cache(self, input_tensor: torch.Tensor) -> None:
        self._cached = input_tensor * 2

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._cached is None:
            self._compute_and_cache(input_tensor)
        return self._cached

    def clear_cache(self) -> None:
        """Utility to clear the cache (not used below)."""
        self._cached = None


model = RefactoredCacheLayer()
x = torch.randn(10, requires_grad=True)
loss = model(x).sum()

# First backward pass
loss.backward()

# Second backward pass (causes error)
loss.backward()