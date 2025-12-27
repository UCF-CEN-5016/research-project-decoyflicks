import torch
from vector_quantize_pytorch import ResidualSimVQ

# Create the ResidualSimVQ instance with corrected parameters
quantizer = ResidualSimVQ(
    dim=1024,
    codebook_size=1024,
    num_heads=8,
    use_dropout=True,
    channels_first=True
)

# Sample data (2 samples, 17 timesteps, 1024 dim)
x = torch.randn(2, 17, 1024)

# Forward pass to trigger the bugs
try:
    quantized, indices, loss = quantizer(x)
except Exception as e:
    print(f"Error 1 (undefined variable): {e}")

# Inspect shapes when running without channels_first
quantizer.channels_first = False
quantized, indices, all_losses, all_indices = quantizer(x, return_all_losses=True)

print("\nShape inconsistencies:")
print("all_losses shapes:", [l.shape for l in all_losses])
print("all_indices shapes:", [i.shape for i in all_indices])