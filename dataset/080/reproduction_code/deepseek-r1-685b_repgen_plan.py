import torch
from vit_pytorch.vit_for_small_dataset import ViT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Bug Type: Validation Accuracy Higher Than Training Accuracy
# Bug Description: Validation accuracy consistently exceeds training accuracy during ViT training

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

# Define transformations and create a fake dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=2, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the ViT model
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

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total

# Training loop
for epoch in range(5):
    model.train()
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        correct, total = calculate_accuracy(outputs, labels)
        train_correct += correct
        train_total += total
    
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct, total = calculate_accuracy(outputs, labels)
            val_correct += correct
            val_total += total
    
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")
    # Typically shows val_acc > train_acc