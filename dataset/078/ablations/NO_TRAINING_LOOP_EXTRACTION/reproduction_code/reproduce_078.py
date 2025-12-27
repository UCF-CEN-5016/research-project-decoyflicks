import torch
import torch.nn as nn
from vit_pytorch.vit import ViT
from vit_pytorch.simple_vit import SimpleViT
from vit_pytorch.mae import MAE
from vit_pytorch.dino import Dino

torch.manual_seed(42)

batch_size = 5
feature_dim = 1024
nested_tensor_input = torch.randn(batch_size, 2, feature_dim)

model = ViT()  # Assuming ViT is the navit_nest_3d model
loss_fn = nn.MSELoss()
j2 = 3
target_tensor = torch.randn(batch_size, j2, feature_dim)

output = model(nested_tensor_input)
loss = loss_fn(output, target_tensor)

try:
    loss.backward()
    gradient_shape = model.parameters().__next__().grad.shape
    assert gradient_shape == (batch_size, 2, feature_dim), f"Unexpected gradient shape: {gradient_shape}"
    print(f"Gradient shape: {gradient_shape}")
except Exception as e:
    print(f"Error during backward pass: {e}")