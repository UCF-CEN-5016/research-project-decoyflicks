import torch
from typing import Tuple

_EPSILON = 1e-6

def _ensure_nonzero_positions(positions: torch.Tensor) -> torch.Tensor:
    """
    Convert positions to float and add a small epsilon to avoid division by zero.
    """
    return positions.float() + _EPSILON

def apply_positional_division(queries: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Divide queries and keys by the (possibly adjusted) positions tensor.
    The positions tensor is reshaped to (1, 1, -1) before broadcasting.
    """
    safe_positions = _ensure_nonzero_positions(positions)
    broadcast_shape = safe_positions.view(1, 1, -1)
    return queries / broadcast_shape, keys / broadcast_shape

if __name__ == "__main__":
    # Example tensors
    queries = torch.randn(1, 10, 64)
    keys = torch.randn(1, 10, 64)
    positions = torch.tensor([0, 1, 2, 3, 4])  # contains zero values

    # Apply positional division
    queries_divided, keys_divided = apply_positional_division(queries, keys, positions)

    # Count NaNs in the divided keys tensor
    nan_count = torch.isnan(keys_divided).sum()
    print(f"Number of NaNs in K_rot: {nan_count}")