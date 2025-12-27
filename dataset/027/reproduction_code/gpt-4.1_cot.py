import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal Discriminator adapted from DCGAN tutorial,
# but without final adaptive pooling, so output spatial dims are > 1 for 128x128 input
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
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

            # Without next downsample - output will be 8x8 instead of 1x1
            # Remove final conv that would reduce to 1x1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # output shape will be (batch,1,5,5)
        ) 

    def forward(self, input):
        return self.main(input)

# Initialize model and loss
D = Discriminator()
criterion = nn.BCELoss()

batch_size = 128
nc = 3
image_size = 128

# Fake batch of images
x = torch.randn(batch_size, nc, image_size, image_size)

# Forward pass through discriminator
output = D(x)  # shape: (batch, 1, h, w)
print("Discriminator output shape:", output.shape)

# Flatten output to 1D tensor
output_flat = output.view(-1)  # (batch * h * w)

# Labels with shape (batch_size) only
labels = torch.ones(batch_size)

print("Output flat shape:", output_flat.shape)
print("Labels shape:", labels.shape)

# This will raise the error:
try:
    loss = criterion(output_flat, labels)
except Exception as e:
    print("Error:", e)