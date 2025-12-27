import torch
from vector_quantize_pytorch import ResidualSimVQ

# Setup with problematic parameters
quantizer = ResidualSimVQ(
    dim = 1024,
    codebook_size = 1024,
    heads = 8,
    quantize_dropout = True,  # Triggers the undefined return_loss
    channels_first = True     # Leads to shape inconsistencies
)

# Sample data (2 samples, 17 timesteps, 1024 dim)
x = torch.randn(2, 17, 1024)

# Forward pass that triggers both bugs
try:
    quantized, indices, loss = quantizer(x)
except NameError as e:
    print(f"Error 1 (undefined variable): {e}")

# Inspect shapes when running without channels_first
quantizer.channels_first = False
quantized, indices, all_losses, all_indices = quantizer(x, return_all_losses=True)

print("\nShape inconsistencies:")
print("all_losses shapes:", [l.shape for l in all_losses])
print("all_indices shapes:", [i.shape for i in all_indices])