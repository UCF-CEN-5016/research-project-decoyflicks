import torch

class RotaryPositionalEmbedding:
    def __init__(self, dim, max_len=100):
        self.cos_cached = torch.zeros(max_len, dim)
        self.sin_cached = torch.zeros(max_len, dim)
        for i in range(max_len):
            self.cos_cached[i] = torch.cos(torch.arange(0, dim) + i)
            self.sin_cached[i] = torch.sin(torch.arange(0, dim) + i)

    def forward(self, x):
        # Simplified example assuming x_rope and neg_half_x are derived from x
        # For demonstration, let's assume x_rope and neg_half_x are x with some operation
        x_rope = x.clone()
        neg_half_x = -0.5 * x
        
        # Incorrect operation that would cause the bug
        # x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        
        # Corrected operation
        d = 3  # Assuming this is the dimension we want to slice to
        x_rope = (x_rope * self.cos_cached[:x.shape[0], :d]) + (neg_half_x * self.sin_cached[:x.shape[0], :d])
        
        return x_rope

# Setup
dim = 4  # Original dimension
max_len = 10
x = torch.randn(3, dim)  # Example tensor

# Create an instance of RotaryPositionalEmbedding
rpe = RotaryPositionalEmbedding(dim, max_len)

# Apply the forward pass
result = rpe.forward(x)

print("Result shape:", result.shape)