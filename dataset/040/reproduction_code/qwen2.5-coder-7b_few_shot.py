import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class SelfAttention(nn.Module):
    """
    Simple self-attention module that computes scaled dot-product attention.
    Expects input of shape (batch, seq_len, dim) and returns attention
    weights of shape (batch, seq_len, seq_len).
    """
    def __init__(self, embedding_dim: int) -> None:
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self._scale = math.sqrt(self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compute_attention(x)

    def _compute_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, dim)
        scores = torch.matmul(inputs, inputs.transpose(1, 2)) / self._scale
        attention_weights = F.softmax(scores, dim=1)
        return attention_weights

# Test with a sample input
if __name__ == "__main__":
    model = SelfAttention(embedding_dim=64)
    input_tensor = torch.randn(1, 8, 64)  # batch=1, seq_len=8, dim=64
    output = model(input_tensor)
    print("Attention output shape:", output.shape)
    print("Attention values:\n", output)