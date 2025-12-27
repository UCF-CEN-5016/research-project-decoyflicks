import torch
from transformers import xformers
from einops import rearrange


def create_input(batch_size: int = 1, features: int = 32) -> torch.Tensor:
    return torch.randn(batch_size, features)


def apply_rotary_xpos(tensor: torch.Tensor) -> torch.Tensor:
    rotary = xformers.RotaryXPos()
    return rotary(tensor)


def reshape_to_column(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, 'n -> n 1')


def main() -> None:
    input_tensor = create_input()
    rotated = apply_rotary_xpos(input_tensor)
    result = reshape_to_column(rotated)

    print("Input shape:", input_tensor.shape)
    print("Output shape after rearrangement:", result.shape)


if __name__ == "__main__":
    main()