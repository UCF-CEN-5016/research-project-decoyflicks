import torch
import einops as e

# Import RotaryEmbedding from the module
from your_module import RotaryEmbedding  # Replace with actual module path

# Set seed for reproducibility
torch.manual_seed(42)

# Define input tensor dimensions
sequence_length = 100
hidden_dimension = 768
batch_size = 32

# Create a batched input tensor 't' with random values between -1 and 1
t = torch.rand((batch_size, sequence_length, hidden_dimension)) * 2 - 1

# Create an instance of RotaryEmbedding
rotary_emb = RotaryEmbedding(hidden_dimension)

# Invoke the forward method on the input tensor 't'
freqs = rotary_emb(t)

# Capture the output frequencies
assert freqs.shape == (batch_size, sequence_length, hidden_dimension * 2), "Incorrect shape for freqs"

# Store a copy of the original cache for scales if it exists before invoking the get_scale method
original_cache_scales = rotary_emb.cached_scales.clone() if hasattr(rotary_emb, 'cached_scales') else None

# Call the get_scale method with parameters seq_len=100 and offset=0
scale = rotary_emb.get_scale(t, seq_len=sequence_length, offset=0)

# Capture the output scale
assert scale.shape == (batch_size, sequence_length, hidden_dimension), "Incorrect shape for scale"

# Compare the captured original cache for scales with the current state to determine if the cache has been modified
if original_cache_scales is not None:
    assert torch.equal(rotary_emb.cached_scales, original_cache_scales), "Cache for scales has been modified"
else:
    assert not hasattr(rotary_emb, 'cached_scales'), "Cache for scales does not exist"

print("Test passed successfully!")