import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.cos_cached = torch.cos(torch.arange(max_seq_len)[:, None] * (-1 if True else 1))
        self.sin_cached = torch.sin(torch.arange(max_seq_len)[:, None] * (-1 if True else 1))

    def forward(self, x):
        # Simulating the problematic operation
        x_rope = x.clone()
        neg_half_x = x_rope.clone()
        
        # This line should trigger the error (assuming incorrect indexing)
        # Corrected line is commented below for reference
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        # Corrected line:
        # x_rope = (x_rope * self.cos_cached[:, :, :, :x_rope.shape[0]]) + (neg_half_x * self.sin_cached[:, :, :, :x_rope.shape[0]])
        
        return x_rope

# Minimal setup to reproduce the error
if __name__ == "__main__":
    # Create a model instance with specific dimensions
    model = RotaryPositionalEmbeddings(dim=4, max_seq_len=10)
    
    # Create input tensor
    x = torch.randn(3, 4)  # Assuming this triggers the bug due to dimension mismatch
    
    try:
        # Attempt to run the model
        output = model(x)
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")