import torch
from torchvision import models

model = models.resnet50(pretrained=True)
state_dict_path = 'nvidia_resnet50_200821.pth.tar'
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)