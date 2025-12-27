import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2))
        
        # Initialize weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

# Model and optimizer
model = STN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # High learning rate

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop that produces NaN
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(loader):
        data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)  # Reconstruction loss
        loss.backward()
        optimizer.step()
        
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch}, batch {batch_idx}")
            break
        print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item()}")