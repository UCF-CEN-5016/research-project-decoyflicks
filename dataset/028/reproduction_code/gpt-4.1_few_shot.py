import torch
from torchvision.models.vision_transformer import vit_b_16

# Create vision transformer model for CIFAR10 dataset
# but do not override num_classes (default=16 instead of 10)
model = vit_b_16()

# CIFAR10 has 10 classes
assert model.heads.head.out_features != 10, "Default num_classes should be 10 but got different value"

# Sample input tensor (batch_size=2, 3 color channels, 32x32 image)
x = torch.randn(2, 3, 32, 32)

# Forward pass: output shape mismatch if num_classes != 10
outputs = model(x)
print(f"Model output shape: {outputs.shape}")  # Expected (2, 10), but will be (2, 16)