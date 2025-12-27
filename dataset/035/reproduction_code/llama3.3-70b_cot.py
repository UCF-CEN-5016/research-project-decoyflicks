import torch
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of RotaryPositionalEmbeddings with the incorrect number of features
rotary_pe = RotaryPositionalEmbeddings(3)

# Create a tensor with 4 features to trigger the error
x = torch.randn(1, 10, 4)

# Add triggering conditions
try:
    # Call the method that triggers the error
    rotary_pe(x)
except Exception as e:
    print(f"Error: {e}")