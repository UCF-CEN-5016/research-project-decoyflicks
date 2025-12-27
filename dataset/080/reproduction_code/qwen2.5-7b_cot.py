import torch
import torchvision
import torchvision.transforms as transforms
from vit_pytorch import ViT
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
def load_dataset(data_path, transform):
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    return dataset

train_dataset = load_dataset('path/to/train_data', transform)
val_dataset = load_dataset('path/to/val_data', transform)

# Create data loaders
def create_data_loader(dataset, batch_size, shuffle):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

train_loader = create_data_loader(train_dataset, batch_size=64, shuffle=True)
val_loader = create_data_loader(val_dataset, batch_size=64, shuffle=False)

# Initialize ViT model
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
)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
train_accuracies = []
val_accuracies = []

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

for epoch in range(10):  # 10 epochs
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_accuracy = calculate_accuracy(outputs, labels)
    train_accuracies.append(train_accuracy)

    with torch.no_grad():
        model.eval()
        correct_val = 0
        total_val = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            val_accuracy = calculate_accuracy(outputs, labels)
            val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}: '
          f'Train Accuracy: {train_accuracy:.4f} | '
          f'Val Accuracy: {val_accuracy:.4f}')

# Plotting the results
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()