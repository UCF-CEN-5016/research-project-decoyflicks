import torch
from vit_pytorch.vit_for_small_dataset import ViT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
image_size = 224
num_classes = 2

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Create a fake dataset
dataset = datasets.FakeData(size=1000, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model Configuration
model_config = {
    'image_size': image_size,
    'patch_size': 16,
    'num_classes': num_classes,
    'dim': 1024,
    'depth': 6,
    'heads': 16,
    'mlp_dim': 2048,
    'dropout': 0.1,
    'emb_dropout': 0.1
}

# Initialize the model
model = ViT(**model_config).to(device)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    # Training
    model.train()
    train_correct, train_total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_total += labels.size(0)
        train_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_total += labels.size(0)
            val_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
    # Typically shows val_acc > train_acc