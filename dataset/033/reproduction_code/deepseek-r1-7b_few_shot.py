import torch
from se3transformer import SE3TransformerPooled

# Create sample input with multiple node feature types
batch_size = 8910
input_features = {
    '0': torch.randn(batch_size, 5),
    '1': torch.randn(batch_size, 1)
}

# Define model with properly split features
model = SE3TransformerPooled(
    fiber_in=Fiber({0:5, 1:1}),
    ...
)

# Process input without dimension issues (simulates the problem setup)
processed_input = {k: v.unsqueeze(-1) for k, v in input_features.items()}

# The model processes correctly with split features
output = model(processed_input)
print(f"Output shape: {output.shape}")