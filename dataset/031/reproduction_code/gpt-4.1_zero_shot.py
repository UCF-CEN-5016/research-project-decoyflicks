import torch
import torchvision.models as models

state_dict = torch.load('nvidia_resnet50_200821.pth.tar', map_location='cpu')['state_dict']
model = models.resnet50()
model.load_state_dict(state_dict)