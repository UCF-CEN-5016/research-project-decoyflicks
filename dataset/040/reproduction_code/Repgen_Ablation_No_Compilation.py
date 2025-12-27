import torch

# Assuming attn is defined somewhere in your codebase
attn = torch.randn(10, 10, requires_grad=True)

# Modify this line as suggested in the bug report
attn = attn.softmax(dim=2)