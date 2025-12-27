import torch
import torch.nn as nn
import torch.nn.functional as F

def build_localization_network():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=7),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(8, 10, kernel_size=5),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
    )

def build_localization_fc():
    return nn.Sequential(
        nn.Linear(10 * 3 * 3, 32),
        nn.ReLU(),
        nn.Linear(32, 6)
    )

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.loc_net = build_localization_network()
        self.loc_regressor = build_localization_fc()

    def forward(self, x):
        loc_feats = self.loc_net(x)
        loc_feats = loc_feats.view(-1, 10 * 3 * 3)
        theta = self.loc_regressor(loc_feats)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.transformer = SpatialTransformer()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.transformer(x)
        x = x.view(-1, 16 * 6 * 6)
        x = self.classifier(x)
        return x

# Training setup
model = MNISTModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy data
inputs = torch.randn(32, 1, 28, 28)
targets = torch.randint(0, 10, (32,))

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")