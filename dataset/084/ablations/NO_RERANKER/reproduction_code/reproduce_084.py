import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from vit_pytorch import ViT, MPP  # Assuming the necessary classes are imported from vit_pytorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 128
image_size = 224
patch_size = 32
num_classes = 2
depth = 12
heads = 8
k = 64

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('path/to/dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

transformer = MPP(transformer=ViT(dim=dim, depth=depth, heads=heads, mlp_dim=dim*4, channels=3), 
                  patch_size=patch_size, dim=dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    transformer.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = transformer(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    transformer.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = transformer(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    
    assert accuracy == 100, f'Accuracy is not 100% after epoch {epoch+1}, it is {accuracy:.2f}%'