import torch
import torchvision.transforms as T
from torch.nn.functional import mse_loss

# Define batch size and image dimensions
batch_size = 32
image_size = 224

# Create random input data
input_data = torch.rand((batch_size, image_size, image_size, 3), dtype=torch.float32)

# Initialize Dino model with ResNet-50 as the base network
from torchvision.models import resnet50
from dino import Dino

student_temp = None
teacher_temp = None
augment_fn = None
augment_fn2 = None

net = resnet50(pretrained=True)
dino_model = Dino(net, image_size, student_temp=student_temp, teacher_temp=teacher_temp, augment_fn=augment_fn, augment_fn2=augment_fn2)

# Forward pass
loss = dino_model(input_data, return_embedding=False, return_projection=True, student_temp=student_temp, teacher_temp=teacher_temp)