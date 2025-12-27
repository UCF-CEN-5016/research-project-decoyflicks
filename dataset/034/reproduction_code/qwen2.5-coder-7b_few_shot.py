import torch
import torch.nn.functional as F

def _get_activation_fn(approximate: str):
    """
    Return the activation function corresponding to the given approximate method.

    Supported methods:
    - 'tanh': Gaussian Error Linear Unit (GELU) via torch.nn.functional.gelu
    - 'relu': Rectified Linear Unit via torch.nn.functional.relu
    """
    if approximate == 'tanh':
        return F.gelu
    if approximate == 'relu':
        return F.relu
    raise ValueError(f"Unknown approximate method: {approximate}")

def gelu(x: torch.Tensor, approximate: str = 'tanh') -> torch.Tensor:
    """
    Apply the specified activation to the input tensor.

    Parameters:
    - x: input tensor
    - approximate: 'tanh' for GELU (default) or 'relu' for ReLU

    Returns:
    - Tensor after applying the chosen activation.
    """
    activation_fn = _get_activation_fn(approximate)
    return activation_fn(x)

input_tensor = torch.randn(2, 3)
gelu(input_tensor, approximate='tanh')