import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from efficient_transformer import Linformer
from vit_pytorch import ViT

# Set device to CPU or GPU (change according to your setup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset (use a dummy dataset for simplicity)
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FakeData(size=313, transform=transform)

# Create a sample dataloader
batch_size = 32
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the ViT model with Linformer as the transformer
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# Set the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model for 5 epochs
for epoch in range(5):
    running_loss = 0.0
    correct_count = 0

    for i, batch in enumerate(data_loader):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_count += (predicted == labels).sum().item()

    accuracy = correct_count / len(dataset)
    print(f"Epoch {epoch+1} - Loss: {running_loss/(i+1)} - Accuracy: {accuracy:.4f}")