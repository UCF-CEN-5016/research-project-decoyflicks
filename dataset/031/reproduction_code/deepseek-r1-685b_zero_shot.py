import torch
import torchvision.models as models

model = models.resnet50(pretrained=False)
state_dict = torch.load('nvidia_resnet50_200821.pth.tar')['state_dict']
model.load_state_dict(state_dict)