import torch
from einops import EinopsError, rearrange

def rearrange_2d_to_1d(input_tensor, pattern):
    if len(input_tensor.shape) != 2:
        raise EinopsError("Input tensor must be 2D for rearranging to 1D")
    return rearrange(input_tensor, pattern)

if __name__ == '__main__':
    # Create a tensor with shape (1, 32) which is 2D
    x = torch.randn(1, 32)

    try:
        # Attempt to rearrange with pattern 'n -> n 1', which expects a 1D input
        # This will trigger the EinopsError if the input has 2 dimensions
        rearrange_2d_to_1d(x, 'n -> n 1')
    except EinopsError as e:
        print(f"EinopsError: {e}")