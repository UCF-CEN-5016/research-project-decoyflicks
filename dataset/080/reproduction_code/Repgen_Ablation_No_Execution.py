import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from vit_pytorch.vit_for_small_dataset import ViT
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define necessary parameters
batch_size = 64
image_size = 224
patch_size = 16
num_classes = 2
dim = 1024
depth = 6
heads = 16
mlp_dim = 2048
dropout = 0.1
emb_dropout = 0.1
learning_rate = 0.003

# Define transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load and split dataset
train_dataset = ImageFolder(root='path_to_train_data', transform=transform)
val_dataset = ImageFolder(root='path_to_val_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create ViT model instance
model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    dropout=dropout,
    emb_dropout=emb_dropout
)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 20
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct_train += (preds == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / len(train_dataset) * 100
    train_accuracies.append(train_accuracy)

    model.eval()
    correct_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            correct_val += (preds == labels).sum().item()

    val_accuracy = correct_val / len(val_dataset) * 100
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

# Plotting the accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracies')
plt.legend()
plt.show()

# Assert that at least one epoch's validation accuracy is higher than the corresponding epoch's training accuracy
assert any(val_acc > train_acc for val_acc, train_acc in zip(val_accuracies, train_accuracies)), "Validation accuracy should not exceed training accuracy."