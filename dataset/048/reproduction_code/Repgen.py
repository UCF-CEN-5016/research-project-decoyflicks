import torch
import numpy as np

def check_tensor_shapes(tensors):
    shapes = [tensor.shape for tensor in tensors]
    return shapes

# Example usage
emissions_arr = [torch.randn(1649), torch.randn(1799)]
shapes = check_tensor_shapes(emissions_arr)
print(shapes)