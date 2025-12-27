import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.fc1 = nn.Linear(512, 256)  # Example layer
        self.fc2 = nn.Linear(256, 16)    # Incorrect num_classes set to 16

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def main():
    torch.manual_seed(42)
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = VisionTransformer().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1):
        for data, target in train_loader:
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print(f'Train Loss: {loss.item()}')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data.view(data.size(0), -1))
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    main()