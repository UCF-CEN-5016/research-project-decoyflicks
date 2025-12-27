import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define the STN module
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel, size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 1 * 1, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 6),  # 2 x 3 = 6 parameters for affine transformation
            # nn.Softmax(dim=1)  # Uncomment this to reproduce NaN loss
        )

    def forward(self, x):
        x = self.localization(x)
        x = x.view(-1, 16 * 1 * 1)
        x = self.fc(x)
        x = x.view(-1, 2, 3)  # (batch_size, 2, 3)
        return x

# Define the full model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stn = STN()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(20 * 4 * 4, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        # Apply STN to get affine parameters
        theta = self.stn(x)
        # Generate grid
        grid = F.affine_grid(theta, x.size())
        # Apply spatial transformation
        x = F.grid_sample(x, grid, align_corners=True)
        # Proceed with the rest of the model
        x = self.conv(x)
        x = x.view(-1, 20 * 4 * 4)
        x = self.fc(x)
        return x

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, loss, optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # High learning rate for potential instability

# Training loop
for epoch in range(1):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')