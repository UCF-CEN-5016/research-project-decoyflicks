import torch
from typing import Tuple

def apply_positional_embedding(tokens: torch.Tensor, pos_embedding: torch.Tensor) -> torch.Tensor:
    """
    Add positional embeddings to token representations.

    Args:
        tokens: Tensor of shape (batch_size, num_tokens, d)
        pos_embedding: Tensor of shape (1 or batch_size, num_positions, d)

    Returns:
        Tensor of shape (batch_size, num_tokens, d) equal to tokens plus
        the first num_tokens positional embeddings.
    """
    if tokens.dim() != 3 or pos_embedding.dim() < 2:
        raise ValueError("Expected tokens to be 3D and pos_embedding to be at least 2D tensors.")
    if tokens.size(2) != pos_embedding.size(-1):
        raise ValueError("Token embedding dimension does not match positional embedding dimension.")
    return tokens + pos_embedding[:, : tokens.size(1)]

def _create_example_tensors(batch_size: int, num_patches: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randn(batch_size, num_patches, dim)
    pos_embedding = torch.randn(1, num_patches + 1, dim)
    return tokens, pos_embedding

def main() -> None:
    batch_size = 2
    num_patches = 3
    dim = 64

    tokens, pos_embedding = _create_example_tensors(batch_size, num_patches, dim)
    result = apply_positional_embedding(tokens, pos_embedding)
    print(result.shape)

if __name__ == "__main__":
    main()