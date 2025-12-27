import torch
from typing import Tuple


def make_random_tensor(rows: int, cols: int, dtype=None, device=None) -> torch.Tensor:
    """Create a random tensor with the given shape."""
    return torch.randn(rows, cols, dtype=dtype, device=device)


def align_tensors_to_min_size(t1: torch.Tensor, t2: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim both tensors along `dim` to the minimum size between them."""
    size1 = t1.size(dim)
    size2 = t2.size(dim)
    min_size = min(size1, size2)

    if dim == 0:
        t1_aligned = t1[:min_size, ...]
        t2_aligned = t2[:min_size, ...]
    elif dim == 1:
        t1_aligned = t1[:, :min_size]
        t2_aligned = t2[:, :min_size]
    else:
        index1 = [slice(None)] * t1.dim()
        index1[dim] = slice(0, min_size)
        index2 = [slice(None)] * t2.dim()
        index2[dim] = slice(0, min_size)
        t1_aligned = t1[tuple(index1)]
        t2_aligned = t2[tuple(index2)]

    return t1_aligned, t2_aligned


def concatenate_aligned(t1: torch.Tensor, t2: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Concatenate two tensors along the specified dimension."""
    return torch.cat([t1, t2], dim=dim)


def main() -> None:
    # Create tensors with mismatched sizes along dim=1
    tensor_a = make_random_tensor(1, 1649)
    tensor_b = make_random_tensor(1, 1799)

    # Align along dim=1 and concatenate along dim=0
    a_aligned, b_aligned = align_tensors_to_min_size(tensor_a, tensor_b, dim=1)
    emissions = concatenate_aligned(a_aligned, b_aligned, dim=0)

    # Verify the shape of the concatenated tensor
    print(emissions.shape)


if __name__ == "__main__":
    main()