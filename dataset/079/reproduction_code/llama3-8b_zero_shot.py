import torch
from vit_pytorch import RegionViT

model = RegionViT()
input_ = torch.randn(1, 3, 224, 224)
output = model(input_)