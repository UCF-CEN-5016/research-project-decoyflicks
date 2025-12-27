import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()
        # Spatial transformer localization-network
        self.loc = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 50),
            nn.ReLU(True),
            nn.Linear(50, 10)
        )

    def stn(self, x):
        xs = self.loc(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # Affine grid and sampling
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        # Transform input
        x = self.stn(x)
        x = x.view(-1, 10 * 3 * 3)
        x = self.fc(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using batch size 64 for demonstration (larger batch sizes increase memory)
    train_loader = DataLoader(
        datasets.MNIST('.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=True)

    model = STNNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)  # High LR triggers instability
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backpropagation can produce NaNs if unstable
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")

if __name__ == "__main__":
    train()