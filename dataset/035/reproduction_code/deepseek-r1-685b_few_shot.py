from labml_nn.transformers.rope import RotaryPositionalEmbeddings
import torch

# Current buggy version (will raise error)
try:
    rotary_pe = RotaryPositionalEmbeddings(3)  # Incorrect feature dimension
    x = torch.randn(1, 4)  # Input with 4 features
    output = rotary_pe(x)   # Will fail due to dimension mismatch
except Exception as e:
    print(f"Error with dim=3: {e}")

# Correct version
rotary_pe = RotaryPositionalEmbeddings(4)  # Correct feature dimension
x = torch.randn(1, 4)  # Input with 4 features
output = rotary_pe(x)   # Works correctly
print(f"Successful output shape: {output.shape}")