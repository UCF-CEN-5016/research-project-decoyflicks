import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
image_size = 128  # Larger size
batch_size = 128
num_epochs = 1
lr = 0.0002

# Define the Discriminator (not adjusted for larger image size)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 4, 0),  # Final layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Dummy data (replace with real data)
real_images = torch.rand(batch_size, 3, image_size, image_size)

# Initialize Discriminator
disc = Discriminator()
criterion = nn.BCELoss()
optimizer = optim.Adam(disc.parameters(), lr=lr)

# Simulate training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = disc(real_images)
    labels = torch.ones(batch_size, 1)  # Real labels
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=4, padding=0),  # Final layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)