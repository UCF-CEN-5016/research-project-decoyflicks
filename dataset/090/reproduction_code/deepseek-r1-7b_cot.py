import torch
from torch.nn import Module, Linear

class EncoderDecoder(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()
        
        # Initialize encoder layers
        self.encoder = nn.Sequential(
            Linear(input_size, hidden_size),
            ...
        )
        
        # Ensure to_logits layer is initialized properly
        self.to_logits = Linear(hidden_size, output_size)
        self.to_logits.weight.requires_grad_(True)  # Ensure gradients are computed
        
    def forward(self, x):
        encoded = self.encoder(x)
        return self.to_logits(encoded)