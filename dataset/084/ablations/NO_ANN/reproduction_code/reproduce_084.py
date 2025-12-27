import torch
import torch.nn as nn
import torchvision.transforms as transforms
from vit_pytorch.ats_vit import ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 224
patch_size = 32
num_classes = 2
dim = 128
depth = 12
heads = 8
mlp_dim = 512

model = ViT(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
    channels=3
).to(device)

dataset = torch.randn(1000, 3, 224, 224)
labels = torch.randint(0, 2, (1000,))
train_data, val_data = dataset[:800], dataset[800:]
train_labels, val_labels = labels[:800], labels[800:]

train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(list(zip(val_data, val_labels)), batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == lbls).sum().item()
    
    train_accuracy = correct / len(train_loader.dataset)
    assert train_accuracy > 0.99, "Training accuracy did not exceed 0.99"
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Accuracy: {train_accuracy}')

model.eval()
val_correct = 0
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        val_correct += (outputs.argmax(dim=1) == lbls).sum().item()

val_accuracy = val_correct / len(val_loader.dataset)
assert val_accuracy == 1.0, "Validation accuracy is not 1.0"
print(f'Final Training Accuracy: {train_accuracy}, Final Validation Accuracy: {val_accuracy}')