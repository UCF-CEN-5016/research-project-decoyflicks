import torch
import torch.nn.functional as F

# Trigger the bug by passing a boolean to the 'approximate' parameter
# which expects a string ('tanh' or 'relu') as per PyTorch documentation
try:
    output = F.gelu(torch.tensor([1.0]), approximate=True)
    print(output)
except TypeError as e:
    print(f"Error: {e}")