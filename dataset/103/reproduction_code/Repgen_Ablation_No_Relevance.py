import torch

# Set batch size and input dimensions
batch_size = 1
height = 32
width = 32
dim = 512

# Create random uniform input data
input_data = torch.randn(batch_size, height, width, dim)

# Call VectorQuantizer function with num_embeddings=128 and embedding_dim=256
num_embeddings = 128
embedding_dim = 256

# Assuming the main file has the necessary imports and the VectorQuantizer class
from vector_quantizers import VectorQuantizer

vq = VectorQuantizer(num_embeddings, embedding_dim)

# Verify output contains NaN values in loss calculation
output, indices, commit_loss = vq(input_data)
assert torch.isnan(commit_loss).item()

# Monitor GPU memory usage
pre_memory_allocated = torch.cuda.memory_allocated()
output, indices, commit_loss = vq(input_data)
post_memory_allocated = torch.cuda.memory_allocated()

# Assert GPU memory exceeds expected threshold (e.g., 100MB)
expected_threshold = 100 * 1024 * 1024  # 100MB in bytes
assert post_memory_allocated - pre_memory_allocated > expected_threshold