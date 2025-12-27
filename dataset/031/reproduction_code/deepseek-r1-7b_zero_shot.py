import torch
from torchvision import models

model = models.resnet50()
weights_path = 'https://download.pytorch.org resnet50-19c8e4b8.pth'

# Create a mapping for layer names to adjust due to version differences
layer_mapping = {
    # Remap layers where structure has changed
    r'layers\.(\d+)' : lambda x: f'layer.{x.split(".")[-2]}'
}

loaded_weights = torch.load(weights_path)

for key in list(loaded_weights.keys()):
    new_key = layer_mapping.get(key)
    if new_key is not None:
        loaded_weights[new_key] = loaded_weights.pop(key)

model.load_state_dict(loaded_weights, strict=False)