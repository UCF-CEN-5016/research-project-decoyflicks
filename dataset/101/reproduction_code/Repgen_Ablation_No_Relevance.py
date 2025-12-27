import torch
from typing import Tuple

# Define batch size and sequence length
batch_size = 32
seq_len = 100
dim = 512

# Create random input tensors Q and K
Q = torch.rand(batch_size, seq_len, dim)
K = torch.rand(batch_size, seq_len, dim)

# Instantiate RotaryEmbedding class
class RotaryEmbedding:
    def __init__(self, dim: int, use_xpos: bool):
        self.dim = dim
        self.use_xpos = use_xpos

    def rotate_queries_and_keys(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Placeholder for the actual implementation of rotary embedding rotation
        rotated_Q = Q.clone()
        rotated_K = K.clone()  # This line is a placeholder; replace it with the actual rotation logic
        return rotated_Q, rotated_K

rotary_embedding = RotaryEmbedding(dim=dim, use_xpos=True)

# Call rotate_queries_and_keys method
rotated_Q, rotated_K = rotary_embedding.rotate_queries_and_keys(Q, K)

# Check for NaN values in the resulting K tensor
nan_count = torch.isnan(rotated_K).sum().item()
assert nan_count / (batch_size * seq_len) > 0.001, "Expected at least 0.1% of elements to be NaN"

# Monitor GPU memory usage
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
assert info.used > 2 * 1024**3, "GPU memory usage should exceed 2GB"