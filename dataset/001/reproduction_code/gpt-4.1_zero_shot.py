import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.loc = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2,2),
            nn.ReLU(True)
        )
        self.loc_fc1 = nn.Linear(10*3*3, 32)
        self.loc_fc2 = nn.Linear(32, 6)
        self.loc_fc2.weight.data.zero_()
        self.loc_fc2.bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    def stn(self, x):
        xs = self.loc(x)
        xs = xs.view(-1, 10*3*3)
        theta = self.loc_fc2(self.loc_fc1(xs))
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STNNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

for epoch in range(20):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if torch.isnan(loss):
            print("NaN loss encountered")
            exit()
        loss.backward()
        optimizer.step()