import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vit_pytorch.vit_for_small_dataset import ViT
import matplotlib.pyplot as plt

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up dataset and data loaders
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root="path_to_train_dataset", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="path_to_test_dataset", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Add triggering conditions
model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=2,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Set up training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_accuracies = []
validation_accuracies = []

for epoch in range(10):
    model.train()
    total_correct = 0
    total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    training_accuracy = total_correct / total_samples
    training_accuracies.append(training_accuracy)

    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    validation_accuracy = total_correct / total_samples
    validation_accuracies.append(validation_accuracy)
    print(f"Epoch {epoch+1}, Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

# Plot training and validation accuracy
plt.plot(training_accuracies, label="Training Accuracy")
plt.plot(validation_accuracies, label="Validation Accuracy")
plt.legend()
plt.show()