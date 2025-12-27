import torch
from vector_quantize_pytorch import ResidualSimVQ

class Quantizer:
    def __init__(self, dim, codebook_size, heads, quantize_dropout=False, channels_first=True):
        self.quantizer = ResidualSimVQ(dim=dim, codebook_size=codebook_size, heads=heads, quantize_dropout=quantize_dropout, channels_first=channels_first)

    def forward(self, x, return_all_losses=False):
        if return_all_losses:
            return self.quantizer(x, return_all_losses=True)
        else:
            try:
                return self.quantizer(x)
            except NameError as e:
                print(f"Error 1 (undefined variable): {e}")

# Setup with corrected parameters
quantizer = Quantizer(dim=1024, codebook_size=1024, heads=8, quantize_dropout=False, channels_first=True)

# Sample data (2 samples, 17 timesteps, 1024 dim)
x = torch.randn(2, 17, 1024)

# Forward pass that triggers both bugs
quantized, indices, loss = quantizer.forward(x)

# Inspect shapes when running without channels_first
quantizer.quantizer.channels_first = False
quantized, indices, all_losses, all_indices = quantizer.forward(x, return_all_losses=True)

print("\nShape inconsistencies:")
print("all_losses shapes:", [l.shape for l in all_losses])
print("all_indices shapes:", [i.shape for i in all_indices])