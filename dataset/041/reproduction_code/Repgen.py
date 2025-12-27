import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torch.utils.data import DataLoader
from dalle_pytorch import DiscreteVAE, Adam, ExponentialLR

# Set up wandb for logging
wandb.init(project="discrete-vae-training")

# Prepare sample data
def load_sample_data():
    transform = Compose([
        Resize((256, 256)),
        CenterCrop(256),
        ToTensor()
    ])
    dataset = ImageFolder(root='path_to_dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

# Load data
dataloader = load_sample_data()

# Initialize DiscreteVAE model
model = DiscreteVAE(num_tokens=1024, smooth_l1_loss=True, num_resnet_blocks=2)

# Set up optimizer and scheduler
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Training function
def train_epoch(dataloader, model, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        data = data[0].to(device)  # Assuming 'device' is defined elsewhere (e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        loss = model(data, mask=data.bool())
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device for model and data
model.to(device)  # Move model to device

for epoch in range(5):
    avg_loss = train_epoch(dataloader, model, optimizer, scheduler, epoch)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# Sample batch for testing
sample_batch = next(iter(dataloader))
print(sample_batch.shape)

# This should raise a TypeError if the 'mask' argument is not handled in DiscreteVAE
train_epoch(dataloader, model, optimizer, scheduler, epoch)
wandb.finish()