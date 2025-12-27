import torch
from transformers import ViTForImageClassification
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Minimal setup
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

# Data preparation
transform = transforms.Compose([transforms.Resize(image_size), 
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Model setup
model = ViTForImageClassification.from_pretrained("vit-base", num_classes=num_classes, 
                                                   image_size=image_size, patch_size=patch_size, 
                                                   dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, 
                                                   dropout=dropout, emb_dropout=emb_dropout)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_accuracy = []
val_accuracy = []

for epoch in range(10):
    model.train()
    total_correct = 0
    for batch in dataset:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

    train_accuracy.append(total_correct / len(dataset))

    model.eval()
    val_total_correct = 0
    with torch.no_grad():
        for batch in val_dataset:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total_correct += (predicted == labels).sum().item()

    val_accuracy.append(val_total_correct / len(val_dataset))

# Plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend()
plt.show()