import torch
import torch.nn.functional as F

def gelu(x, approximate=True):
    return F.gelu(x, approximate=approximate)

batch_size = 8
input_tensor = torch.randn(batch_size, 1024)

output = gelu(input_tensor)

# Modify the gelu function to trigger the TypeError
def gelu_modified(x, approximate=False):  # Change to bool
    return F.gelu(x, approximate=approximate)

try:
    output = gelu_modified(input_tensor)
except TypeError as e:
    print(f"Caught an error: {e}")