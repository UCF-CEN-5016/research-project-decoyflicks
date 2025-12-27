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
train_dataset = torchvision.datasets.ImageFolder(
    root='path/to/train_data',
    transform=transform
)
val_dataset = torchvision.datasets.ImageFolder(
    root='path/to/val_data',
    transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False
)

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

# Training loop (without model.train())
train_accuracies = []
val_accuracies = []

for epoch in range(10):  # 10 epochs
    model.eval()  # Model is in eval mode during training (bug)
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = correct_val / total_val
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
plt.title('Training vs Validation Accuracy (Bug Reproduction)')
plt.show()