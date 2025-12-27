import torch
import torch.nn as nn
from torch import Tensor

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # Simplified rotation matrix initialization
        self.rotation_matrix = nn.Parameter(torch.randn(d, d))
        
    def rotate(self, x: Tensor):
        """Apply rotation to input tensor"""
        print("Applying rotation")  # Debug print to show rotations
        return x @ self.rotation_matrix

class ValueWithRoPE(nn.Module):
    def __init__(self, d_model: int, d_values: int):
        super().__init__()
        self.d_values = d_values
        # Value projection
        self.value = nn.Linear(d_model, d_values)
        # Position embeddings
        self.value_pos = RotaryPositionalEmbeddings(d_values)
        
    def forward(self, x: Tensor):
        # Project to value space
        v = self.value(x)
        
        # First rotation (line 230 in original)
        v = self.value_pos.rotate(v)
        
        # Second rotation (line 234 in original)
        v = self.value_pos.rotate(v)
        
        return v

# Test the implementation
d_model = 64
d_values = 32
batch_size = 2
seq_len = 10

model = ValueWithRoPE(d_model, d_values)
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass will print "Applying rotation" twice
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")