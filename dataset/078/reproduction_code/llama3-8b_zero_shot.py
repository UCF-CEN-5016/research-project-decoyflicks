import torch
from vit_pytorch import NavitNest3D

navit_nest_3d = NavitNest3D()
input_tensor = torch.randn(5, 2, 1024)
output = navit_nest_3d(input_tensor)

output.sum().backward()