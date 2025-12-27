import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
image_size = 128  # Changed from 64 to 128
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Generator feature map size
ndf = 64  # Discriminator feature map size
nc = 3    # Number of channels in the training images (CelebA: 3)

# Generator (simplified for image_size=128)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 128x128
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator (original code for 64x64, insufficient for 128x128)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # 5x5 output for 128x128 input (not 1x1)
            # No sigmoid here, will use BCEWithLogitsLoss
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)  # Flatten to (N*output_size), causes mismatch

# Create models
netG = Generator()
netD = Discriminator()

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

# Fake batch size
batch_size = 4

# Generate fake images
noise = torch.randn(batch_size, nz, 1, 1)
fake_images = netG(noise)

# Labels
real_label = 1.
fake_label = 0.

# Forward pass through discriminator
output = netD(fake_images)

# Labels have size (batch_size,), output might be larger due to spatial dimensions
labels = torch.full((batch_size,), real_label)

# This line will raise the ValueError because output.size() != labels.size()
loss = criterion(output, labels)

print(f"Output size: {output.size()}, Labels size: {labels.size()}")