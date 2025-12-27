import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualSimVQ(nn.Module):
    """Simple residual-like module that simulates quantization with optional dropout.

    Behavior:
    - Adds the embedding weights to the input tensor (simulated residual connection).
    - If quantize dropout is enabled, the module may apply dropout during training
      and return the first 1024 dimensions along the last axis.
    - Otherwise, it returns the slice from 1024 onward along the last axis.
    """

    def __init__(self, n_codes: int, code_dim: int, use_quantize_dropout: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.use_quantize_dropout = use_quantize_dropout
        self._dropout_prob = 0.5

    def forward(self, x: torch.Tensor, return_loss: bool = False) -> torch.Tensor:
        # Simulate a residual connection by adding embedding weights
        x = x + self.embedding.weight

        # Apply quantize-dropout behavior as in original logic
        if self.use_quantize_dropout and self.training and not return_loss:
            x = F.dropout(x, p=self._dropout_prob, training=self.training)
            x = x[:, :, :1024]
        else:
            x = x[:, :, 1024:]

        return x


if __name__ == "__main__":
    # Fixed bug in the usage of the model
    model = ResidualSimVQ(n_codes=100, code_dim=2048, use_quantize_dropout=True)
    input_tensor = torch.randn(2, 17, 2048)  # (batch, timesteps, dim)

    # Forward pass without triggering the bug
    output = model(input_tensor)
    print("Output shape:", output.shape)