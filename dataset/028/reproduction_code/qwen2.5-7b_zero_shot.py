import torch
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=16):
        super(VisionTransformer, self).__init__()
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.linear(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = VisionTransformer()
print(f"Model num_classes: {model.linear.out_features}")