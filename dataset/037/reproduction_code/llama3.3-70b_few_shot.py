import torch
import math

# Define the RoPE value embedding function
def rope_value_embedding(x, dim, max_seq_len):
    # Calculate the rotation matrix
    freq = torch.arange(0, dim, dtype=torch.float32) / dim
    freq = 1 / (10000 ** (freq))
    
    # Rotate the embedding twice ( suspected bug )
    x = x * freq.unsqueeze(0)
    x = x * freq.unsqueeze(0)  # Second rotation, possibly incorrect
    
    # Apply the rotation to the embedding
    x = torch.cat([x, x], dim=-1)
    
    # Calculate the outer product of the rotation matrix
    freq_outer = torch.einsum('bi,bj->bij', freq, freq)
    x = x * freq_outer.unsqueeze(0)
    
    # Return the rotated embedding
    return x

# Test the function with a sample embedding
dim = 128
max_seq_len = 512
x = torch.randn(1, max_seq_len, dim)

# Apply the RoPE value embedding
embedding = rope_value_embedding(x, dim, max_seq_len)

# Print the resulting embedding
print(embedding.shape)