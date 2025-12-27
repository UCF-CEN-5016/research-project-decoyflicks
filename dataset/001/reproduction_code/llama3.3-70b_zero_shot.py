import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(True),
            nn.Linear(50, 6)
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(True),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 320)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = torch.nn.functional.affine_grid(theta, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d, loss = %.3f' % (epoch+1, running_loss/(i+1)))