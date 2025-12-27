import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 1
nheads = 4
c = 64
h = 10
w = 10

# Create a dummy input tensor with shape [batch_size, nheads, c, h, w]
x = torch.randn(batch_size, nheads, c, h, w)

# Define the Reshape layer
x = x.view(batch_size, nheads, c, h, w)
# Apply ReLU (though in practice this might not be necessary as it's a learnable step)
x = F.relu(x)

# Reshape to apply LayerNorm correctly. This reshapes into [batch_size * nheads, -1]
# where -1 flattens all dimensions except channels.
x = x.view(-1, c)  # Now shape is [batch_size*nheads*c*h*w, ]
layer_norm = nn.LayerNorm(c)

# Apply ReLU again (not essential for repro but shows the flow)
x = F.relu(x)

# This line will cause an error because LayerNorm expects input in a compatible shape
try:
    x = layer_norm(x)
except RuntimeError as e:
    print(f"LayerNorm error: {e}")