import torch
import einops


def create_random_tensor(batch_size: int = 1, feature_dim: int = 32) -> torch.Tensor:
    return torch.randn(batch_size, feature_dim)


def add_leading_dimension(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(0)


def rearrange_to_n_1(tensor: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(tensor, 'n d -> n 1')


def main():
    base_tensor = create_random_tensor()
    expanded_tensor = add_leading_dimension(base_tensor)
    result = rearrange_to_n_1(expanded_tensor)
    return result


if __name__ == "__main__":
    output = main()