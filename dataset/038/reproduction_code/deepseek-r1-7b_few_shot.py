x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

x_rope = ((x_rope * self.cos_cached[:x.shape[0], :, :, :self.d]) +
          (neg_half_x * self.sin_cached[:x.shape[0], :, :, :self.d]))

import torch
from torch.nn import functional as F

class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # Cache the computed cosine and sine for positional embeddings
        self.register_buffer('cos_cached', torch.zeros(max_len, dtype=torch.float32))
        self.register_buffer('sin_cached', torch.zeros(max_len, dtype=torch.float32))
        
    def forward(self, x):
        """
        Apply rotary position encoding to input tensor x.
        Slices the cached cosine and sine values appropriately for each element in x
        across all dimensions to prevent size mismatch errors.
        """
        B, T, C = x.shape  # batch_size, sequence_length, features
        # Slice only up to needed dimension 'd' (features)
        cos_vals = self.cos_cached[:C][:T][:, None, :, :self.d]
        sin_vals = self.sin_cached[:C][:T][:, None, :, :self.d]
        
        x_rot = F.rot90(x * cos_vals + (-x[..., 1:] if T > 1 else x) * sin_vals, k=1)
        return x_rot

# Create a RotaryPositionalEncoding instance with input dimension of 4 (d)
model = RotaryPositionalEncoding(d_model=4)

# Generate random input tensor of shape [32, 10, 8] (batch_size=32, sequence_length=10, features=8)
x = torch.randn(32, 10, 8)

# Apply the model's forward pass
y = model(x)

# The result should be free of dimension mismatch errors and correctly shaped
print(f"Output shape: {y.shape}")