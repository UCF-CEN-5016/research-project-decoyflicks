import torch
import torch.nn.functional as F

def run_gelu_test():
    """Generate a random vector and apply GELU activation (default parameters)."""
    input_tensor = torch.randn(5)
    gelu_output = F.gelu(input_tensor)

if __name__ == "__main__":
    run_gelu_test()