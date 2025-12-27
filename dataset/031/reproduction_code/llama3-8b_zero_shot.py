import torch
from torchvision import models

model = models.resnet50(pretrained=True)
state_dict = torch.load('nvidia_resnet50_200821.pth.tar')['state_dict']
model.load_state_dict(state_dict)

# No need to run inference, just loading the state_dict causes the error