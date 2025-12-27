import torch
from vit_pytorch.regionvit import RegionViT

# Initialize model
model = RegionViT(
    dim = 64,
    depth = 6,
    heads = 8,
    window_size = 7,
    dim_head = 32,
    num_classes = 1000
)

# Create dummy input and target
x = torch.randn(4, 224, 224, 3)
target = torch.randint(0, 1000, (4,))

# Training step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()

# Forward pass
output = model(x)
loss = torch.nn.CrossEntropyLoss()(output, target)

# Backward pass
loss.backward()

# Check gradients of layer norm parameters
layer_norm = model.local_transformer.layers[0].norm1
print(f"LayerNorm weight grad shape: {layer_norm.weight.grad.shape}")
print(f"Expected shape should match normalized dimension")