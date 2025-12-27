from vit_pytorch import CrossViT

# Minimal input setup
import torch

# Create dummy input tensor (batch_size=1, channels=3, height=224, width=224)
x = torch.randn(1, 3, 224, 224)

# Instantiate CrossViT model with default parameters
model = CrossViT()

# Forward pass triggers AttributeError if internal attribute or import is missing/wrong
output = model(x)

print(output.shape)