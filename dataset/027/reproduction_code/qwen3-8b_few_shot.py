import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Discriminator that is not adjusted for 128x128 input
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),    # 8x8 -> 4x4
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a dummy input with shape (batch_size, channels, height, width)
batch_size = 8
input_tensor = torch.rand(batch_size, 3, 128, 128).to(device)

# Initialize the discriminator
disc = Discriminator().to(device)

# Forward pass
output = disc(input_tensor)  # Output shape: (8, 1, 4, 4) = 16 elements per image

# Try to apply BCE loss (which expects a 1D tensor of shape (batch_size, 1))
# This will raise a ValueError due to size mismatch
target = torch.rand(batch_size, 1).to(device)
loss = F.binary_cross_entropy(output.view(batch_size, 1), target)

print(loss)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),    # 8x8 -> 4x4
            nn.AdaptiveAvgPool2d(1),  # Reduce spatial dimensions to 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)