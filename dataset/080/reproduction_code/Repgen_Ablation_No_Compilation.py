import torch
from torchvision import datasets, transforms
from vit_pytorch.vit_for_small_dataset import ViT

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=transform)
val_dataset = datasets.ImageFolder(root='path_to_val_data', transform=transform)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Initialize model
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# Training loop
for epoch in range(10):
    model.train()
    train_loss = 0.0
    correct_train = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
    
    train_acc = correct_train / len(train_dataset)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader.dataset)}, Training Accuracy: {train_acc}')

    model.eval()
    val_loss = 0.0
    correct_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
    
    val_acc = correct_val / len(val_dataset)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader.dataset)}, Validation Accuracy: {val_acc}')