import torch
from fairseq.modules import RelPositionMultiHeadedAttention

# Define batch size and sequence length
batch_size = 2
sequence_length = 10
embed_dim = 512

# Create random input data
input_data = torch.randn(batch_size, sequence_length, embed_dim)

# Initialize the attention mechanism
attention = RelPositionMultiHeadedAttention(
    num_heads=8,
    attention_dim=embed_dim,
    bias=True,
    relative_position_bucket=32
)

# Call the forward method
output = attention(input_data)

# Verify NaN values in output
assert not torch.isnan(output).any(), "Output contains NaN values"

# Monitor memory usage
import tracemalloc
tracemalloc.start()
initial_memory = tracemalloc.get_traced_memory()[0]

# Call the forward method again to increase memory usage
output = attention(input_data)

current_memory, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Assert significant memory increase due to bias initialization
assert peak_memory > initial_memory + 1e6, "Memory usage did not significantly increase"