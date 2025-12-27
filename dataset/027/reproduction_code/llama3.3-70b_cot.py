import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DCGAN discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Define the dataset loader for the CelebA dataset
transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = datasets.ImageFolder(root='./celeba', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define the binary cross-entropy loss function
criterion = nn.BCELoss()

# Initialize the discriminator and optimizer
discriminator = Discriminator(128).to(device)
optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
for epoch in range(10):
    for i, data in enumerate(dataloader):
        # Get the input and label
        input = data[0].to(device)
        label = torch.ones(input.size(0), 1).to(device)

        # Forward pass
        output = discriminator(input)

        # Compute the binary cross-entropy loss
        errD_real = criterion(output.view(-1), label.view(-1))  # Note the view(-1) to flatten the output

        # Backward pass
        optimizer.zero_grad()
        errD_real.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {errD_real.item()}')