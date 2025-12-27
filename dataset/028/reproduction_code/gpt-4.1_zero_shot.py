import torch
from torchvision.models import vit_b_16

model = vit_b_16(num_classes=16)
print("num_classes:", model.heads.head.out_features)