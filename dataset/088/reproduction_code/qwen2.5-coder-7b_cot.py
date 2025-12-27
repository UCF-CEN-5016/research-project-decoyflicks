import torch
from x_transformers import XTransformer
from typing import Tuple


def create_random_input(batch_size: int = 1, seq_length: int = 5) -> torch.Tensor:
    """Create a random input tensor with the given batch and sequence dimensions."""
    return torch.randn(batch_size, seq_length)


def build_transformer() -> XTransformer:
    """Instantiate and return an XTransformer model."""
    return XTransformer()


def align_input_right(input_tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Wrap the transformer and call its align_right method with the provided pad_id."""
    model = build_transformer()
    return model.align_right(input_tensor, pad_id=pad_id)


def main() -> torch.Tensor:
    # Step 1: Create a random input tensor
    batch = 1
    seq_len = 5
    input_tensor = create_random_input(batch, seq_len)

    # Step 2: Define the pad_id that shouldn't be used in padding (e.g., non-zero)
    pad_value = 42

    # Step 3: Wrap the model and call align_right with the specified pad_id
    return align_input_right(input_tensor, pad_id=pad_value)


result = main()