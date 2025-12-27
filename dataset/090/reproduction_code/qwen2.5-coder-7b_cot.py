import torch
import torch.nn as nn
from torch.nn import Module, Linear

class EncoderDecoder(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self._encoder = self._build_encoder(input_size, hidden_size)
        self._to_logits = Linear(hidden_size, output_size)
        # Ensure gradients are enabled for logits layer weights
        self._to_logits.weight.requires_grad_(True)

    def _build_encoder(self, in_features: int, out_features: int) -> nn.Sequential:
        # Minimal encoder: single linear layer mapping input to hidden representation
        return nn.Sequential(Linear(in_features, out_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self._encoder(inputs)
        return self._to_logits(hidden)