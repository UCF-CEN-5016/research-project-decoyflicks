import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from efficient_transformer import Linformer, ViT  # Assume these are available in the environment

# 🧠 Step 1: Create a dummy dataset with identical images and labels
# - 100 random images (3 channels, 224x224)
# - All labels are 0 (binary classification)
images = torch.rand(100, 3, 224, 224)
labels = torch.zeros(100, dtype=torch.long)

dataset = TensorDataset(images, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 🧠 Step 2: Define the ViT model with Linformer
model = ViT(
    image_size=224,
    patch_size=32,
    num_classes=2,
    dim=64,
    depth=6,
    heads=8,
    mlp_dim=128,
    dropout=0.1,
    emb_dropout=0.1
)

# 🧠 Step 3: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🧠 Step 4: Training loop
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 🧠 Step 5: Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy:.2f}%")