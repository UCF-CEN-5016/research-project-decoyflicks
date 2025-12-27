import torch
import numpy as np

def reproduce_gelu_bug():
    x = torch.randn(10)
    return torch.nn.functional.gelu(x, approximate=True)

if __name__ == "__main__":
    try:
        result = reproduce_gelu_bug()
        print(result)
    except TypeError as e:
        print(f"Error: {e}")