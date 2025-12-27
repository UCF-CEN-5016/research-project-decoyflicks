import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128 * 8 * 8)
        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, z):
        z = torch.relu(self.fc1(z)).view(-1, 128, 8, 8)
        z = self.conv_block1(z)
        z = self.conv_block2(z)
        z = self.conv_block3(z)
        return z

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )
        self.fc1 = nn.Linear(128 * 16 * 16, 1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, 128 * 16 * 16)
        x = torch.sigmoid(self.fc1(x))
        return x

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
batch_size = 128
image_size = 128

# Load the dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = torchvision.datasets.CelebA(root='./data', split='train', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define the loss function and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Train the model
for epoch in range(1):
    for i, (images, _) in enumerate(dataloader):
        # Train the discriminator
        images = images.to(device)
        label = torch.ones((images.size(0), 1)).to(device)
        output = discriminator(images)
        errD_real = criterion(output, label)
        z = torch.randn((images.size(0), 100)).to(device)
        fake_images = generator(z)
        label = torch.zeros((images.size(0), 1)).to(device)
        output = discriminator(fake_images)
        errD_fake = criterion(output, label)
        errD = errD_real + errD_fake
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()

        # Train the generator
        z = torch.randn((images.size(0), 100)).to(device)
        fake_images = generator(z)
        output = discriminator(fake_images)
        label = torch.ones((images.size(0), 1)).to(device)
        errG = criterion(output, label)
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()