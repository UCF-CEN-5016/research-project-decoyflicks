import torch
import torch.nn as nn

class RopeValuePE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Simple rotation implementation for demonstration
        self.rotation = nn.Linear(dim, dim)
    
    def forward(self, value_embedding):
        # First rotation (line 230)
        value_embedding = self.rotation(value_embedding)
        
        # Second redundant rotation (line 234)
        value_embedding = self.rotation(value_embedding)
        
        return value_embedding

# Test the buggy implementation
dim = 64
batch_size = 4
seq_len = 16

# Create test input
value_embedding = torch.rand(batch_size, seq_len, dim)

# Create model with redundant rotations
model = RopeValuePE(dim)

# Get output with both rotations
output_with_redundant = model(value_embedding.clone())

# Modify the model to apply only a single rotation
class FixedRopeValuePE(RopeValuePE):
    def forward(self, value_embedding):
        # Apply rotation only once
        value_embedding = self.rotation(value_embedding)
        return value_embedding

fixed_model = FixedRopeValuePE(dim)
# Use same weights for fair comparison
fixed_model.rotation.weight = model.rotation.weight
fixed_model.rotation.bias = model.rotation.bias

# Get output with single rotation
output_fixed = fixed_model(value_embedding.clone())

# Check difference between outputs
difference = (output_with_redundant - output_fixed).abs().mean().item()
print(f"Mean difference: {difference}")
print(f"Redundant rotation detected: {difference > 0}")