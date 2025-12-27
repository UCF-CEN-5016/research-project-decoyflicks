import torch

# Ensure PyTorch version 1.9.0+cu111 is installed
# Install using pip: pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Check if mps backend is available
if hasattr(torch.backends, 'mps'):
    print("MPS backend is available.")
else:
    raise AttributeError("Module 'torch.backends' has no attribute 'mps'")