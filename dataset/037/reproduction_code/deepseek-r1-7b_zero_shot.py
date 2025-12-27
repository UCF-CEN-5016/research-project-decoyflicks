import math

# Define model dimensions
d_model = 512
scale = 0.5

# Create sample value tensor
value = torch.randn(d_model)

# Apply two scalings (double scaling)
value = scale * value
value = math.sqrt(d_model) * value

import math

# Define model dimensions
d_model = 512
scale = 0.5

# Create sample value tensor
value = torch.randn(d_model)

# Apply a single scaling factor (sqrt(d_model))
value = math.sqrt(d_model) * value

import torch
import math

def test_value_embedding():
    d_model = 512
    scale = 0.5
    
    # Create a random tensor for demonstration
    value = torch.randn(d_model)
    
    # Incorrect double rotation (shouldn't have both scalings applied)
    value_rotatedtwice = value * math.sqrt(d_model) * math.sqrt(d_model)
    
    print("Value after incorrect rotations:", value_rotatedtwice)

test_value_embedding()