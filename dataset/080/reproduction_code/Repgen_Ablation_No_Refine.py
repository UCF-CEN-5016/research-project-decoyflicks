import torch
from vit_pytorch.vit_for_small_dataset import ViT
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Define batch size and image dimensions
batch_size = 64
image_size = 224

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define the model
model = ViT(
    image_size=image_size,
    patch_size=16,
    num_classes=10,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# Move the model to a device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    train_loss = 0.0
    correct_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        correct_train += (preds == labels).sum().item()

    train_accuracy = correct_train / len(train_dataset)

    val_loss = 0.0
    correct_val = 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * images.size(0)
            correct_val += (preds == labels).sum().item()

    val_accuracy = correct_val / len(val_dataset)

    print(f'Epoch {epoch+1}/{10}, Train Loss: {train_loss/len(train_loader.dataset)}, Train Acc: {train_accuracy*100:.2f}%, Val Loss: {val_loss/len(val_loader.dataset)}, Val Acc: {val_accuracy*100:.2f}%')

    # Assert that at least once during training, the validation accuracy is higher than the training accuracy
    if val_accuracy > train_accuracy:
        print("Validation accuracy is higher than training accuracy")