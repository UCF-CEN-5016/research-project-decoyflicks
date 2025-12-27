import torch
from vit_pytorch.vit_for_small_dataset import ViT
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Set random seed for reproducibility
torch.manual_seed(42)

# Define batch size and image dimensions
batch_size = 64
image_dim = 224

# Data augmentation transforms
transform = transforms.Compose([
    transforms.Resize((image_dim, image_dim)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = ImageFolder(root='path_to_train_data', transform=transform)
val_dataset = ImageFolder(root='path_to_val_data', transform=transform)

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define ViT model parameters and initialize the model
model_params = {
    "image_size": 224,
    "patch_size": 16,
    "num_classes": 2,
    "dim": 1024,
    "depth": 6,
    "heads": 16,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1
}
model = ViT(**model_params)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for img, label in train_loader:
        optimizer.zero_grad()
        preds = model(img)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in val_loader:
            preds = model(img)
            loss = criterion(preds, label)
            val_loss += loss.item()
            _, predicted = torch.max(preds.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    train_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}, Training Accuracy: {train_acc:.2f}%')