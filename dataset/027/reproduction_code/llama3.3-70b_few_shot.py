import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the DCGAN architecture for a larger image size (128x128)
class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

# Initialize the DCGAN model, loss function, and input tensor
model = DCGAN()
criterion = nn.BCELoss()
input_tensor = torch.randn(1, 3, 128, 128)

# This will cause the error due to mismatched tensor sizes
output = model(input_tensor)
label = torch.ones(128)  # Incorrect label size
errD_real = criterion(output, label)

print(f"Error: {errD_real}")