import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=16):
        super(VisionTransformer, self).__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    model = VisionTransformer()
    model.to(device)
    data, _ = next(iter(trainloader))
    data = data.to(device)
    output = model(data.view(-1, 512))
    print(model.fc.out_features)

if __name__ == "__main__":
    main()