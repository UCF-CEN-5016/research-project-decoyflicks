import torch
import einops

# Define minimal environment
torch.manual_seed(42)  # For reproducibility

# Create a basic tensor
tensor = torch.randn(1, 32)

try:
    # Apply rearrange-reduction pattern that should trigger the bug
    einops.rearrange(tensor, 'n -> n 1')
except einops.EinopsError as e:
    print(f"Caught EinopsError: {e}")