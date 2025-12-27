import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from vit_pytorch import ViT  # Assuming ViT is defined in vit_pytorch module

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 128
image_size = 224
patch_size = 32
num_classes = 2
depth = 12
heads = 8
k = 64

model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads).to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item()
        print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    assert accuracy == 1.0, "Accuracy is not 100%"
    print(f'Epoch {epoch+1} completed with 100% accuracy.')