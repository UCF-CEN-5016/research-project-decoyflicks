import torch
import torch.nn as nn
import torch.optim as optim

# Set up minimal environment
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64, 4, stride=2),  # Input size: 5x5, Output size: 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.main(z.view(-1, 100, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2),  # Input size: 128x128, Output size: 64x64
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Add triggering conditions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100

G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()

# Trigger the bug
real_data = torch.randn(128, 1, 128, 128).to(device)  # Input size: 128x128
label = torch.tensor([1] * 128).unsqueeze(-1).unsqueeze(-1).float().to(device)  # Target size: 128

errD_real = criterion(G(real_data), label)