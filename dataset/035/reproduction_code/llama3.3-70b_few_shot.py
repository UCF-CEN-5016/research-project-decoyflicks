from labml_nn.transformers.rope import RotaryPositionalEmbeddings

# Define the number of features
num_features = 4

# Incorrect initialization with 3 instead of 4
rotary_pe = RotaryPositionalEmbeddings(3)

# This will cause an error when using rotary_pe with num_features
try:
    # Simulate usage of rotary_pe
    rotary_pe(torch.randn(1, num_features))
except Exception as e:
    print(f"Error: {e}")

# Correct initialization
rotary_pe_correct = RotaryPositionalEmbeddings(num_features)

# This will work correctly
rotary_pe_correct(torch.randn(1, num_features))