import torch
from vit_pytorch.na_vit_nested_tensor_3d import NaViT

# Model configuration
model_config = {
    'image_size': 256,
    'patch_size': 32,
    'num_classes': 1000,
    'dim': 1024,
    'depth': 6,
    'heads': 16,
    'mlp_dim': 2048
}

# Instantiate the model
model = NaViT(**model_config).cuda()

# Create nested tensor input
batch_size = 5
images = torch.stack([torch.randn(3, 256, 256).cuda() for _ in range(batch_size)])

# Forward pass
outputs = model(images)

# Create dummy target with correct shape
target = torch.randint(0, 1000, (batch_size,)).cuda()

# Backward pass
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(outputs, target)
loss.backward()

print("Backward pass completed")