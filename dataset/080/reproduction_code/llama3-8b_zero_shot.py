```python
import torch
from torchvision import datasets, transforms
from vit_pytorch.vit_for_small_dataset import ViT

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

transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
val_loss_list = []
accuracy_list = []

for epoch in range(10):
    for inputs, labels in train_dataset:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        total_correct = 0
        val_loss = 0
        for inputs, labels in test_dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(test_dataset.dataset)
        train_loss_list.append(loss.item())
        val_loss_list.append(val_loss / len(test_dataset))
        accuracy_list.append(accuracy)

print(train_loss_list, val_loss_list, accuracy_list)

