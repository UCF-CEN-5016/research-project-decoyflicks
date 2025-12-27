import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

# Load image function
def load_image(image_path):
    return Image.open(image_path).convert('RGB')

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# DataLoader setup
transform = ToTensor()
dataset = CustomDataset(root_dir='path/to/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Neural network model
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(16, 32, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),
    torch.nn.Linear(512, 10)
)

# Initialize model parameters
model.apply(lambda m: next(m.parameters()).data.normal_(0, 0.01))

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss}')

# Verify NaN in loss
assert torch.isnan(epoch_loss).any(), "Loss contains NaN values"
print(f'Final Loss: {epoch_loss}')