import torch
import einops

def create_power_tensor(length: int = 32) -> torch.Tensor:
    """Create a tensor with shape [1, length], containing values 0..length-1."""
    return torch.arange(length).unsqueeze(0)

def try_rearrange(tensor: torch.Tensor, pattern: str) -> None:
    """Attempt to rearrange the tensor with the given einops pattern and print the result."""
    try:
        rearranged = einops.rearrange(tensor, pattern)
        print("Reshaped tensor:", rearranged.shape)
    except Exception as e:
        print("Error:", e)

def main() -> None:
    power = create_power_tensor(32)  # Shape: [1, 32]
    try_rearrange(power, 'n -> n 1')

if __name__ == "__main__":
    main()