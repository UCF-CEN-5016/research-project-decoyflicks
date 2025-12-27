import torch

# Create a dummy input tensor with batch size 2 and sequence length 5 (matching 4 features)
X = torch.randn(2, 5)

# Incorrect usage: Creates an error due to incorrect number of features
rotary_pe_old = RotaryPositionalEmbeddings(3)  # Should be 4

try:
    # This will cause an error with the current number of features
    print("Outputs with incorrect features (3):")
    outputs_old = model(X)
except Exception as e:
    print(f"Error: {e}")

# Correct usage: No error due to correct number of features
rotary_pe_new = RotaryPositionalEmbeddings(4)  # Correct number of features

print("\nOutputs with correct features (4):")
outputs_new = model(X)

# Note that the outputs may differ slightly based on the fixed parameters