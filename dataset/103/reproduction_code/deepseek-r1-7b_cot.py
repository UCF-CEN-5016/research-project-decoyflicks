import torch
from vector_quantize_pytorch.lookup_free_quantization import ResidualLFQ

# Setup environment
device = 'cpu'
batch_size = 1
input_channels = 32
patch_size = (4, 4)
mask_ratio = 0.5

# Create original_input with dimensions that might cause shape issues after reshape
original_input = torch.randn(batch_size, input_channels, 28, 28).to(device)

# Define mask: random binary tensor indicating active elements for loss computation
mask = torch.bernoulli(torch.ones(batch_size, *original_input.shape[1:]) * mask_ratio).to(device)
mask = mask.view(mask.size(0), -1)  # Flatten to match dimensions

# Initialize model with commitment_loss_weight >0 to ensure the bug is triggered
model = ResidualLFQ(
    in_channels=32,
    out_channels=64,
    embedding_dim=16,
    nll_loss=True,
    use_momentum_for_mean=False,
    commit_loss_weight=1.0  # This triggers the bug when >0
)

# Compute forward pass to trigger the bug
try:
    output = model(original_input, mask=mask)
except Exception as e:
    print(f"An error occurred: {e}")