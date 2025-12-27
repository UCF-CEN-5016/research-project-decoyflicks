# Import necessary libraries
import torch
from labml import log
from labml_nn.transformers.rope import RotaryPositionalEmbeddings

def test_rotary_positional_embeddings():
    # Initialize RotaryPositionalEmbeddings with d=4
    rotary_pe = RotaryPositionalEmbeddings(4)
    
    # Create a sample input tensor
    input_tensor = torch.randn(1, 32, 4)
    
    # Apply the RotaryPositionalEmbeddings
    output_tensor = rotary_pe(input_tensor)
    
    # Log the shape of the output tensor to verify
    log.info(f"Output Tensor Shape: {output_tensor.shape}")

# Call the test function
test_rotary_positional_embeddings()