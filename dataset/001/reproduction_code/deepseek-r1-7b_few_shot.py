import torch
from torch import nn
from torch.optim import Adam

class MNISTSpatialNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=5000, num_classes=10):
        super(MNISTSpatialNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model with large layers to trigger memory issues and potential instability
model = MNISTSpatialNetwork(input_size=784, hidden_size=5000, num_classes=10)

# Use Adam optimizer which is more stable for such models
optimizer = Adam(model.parameters(), lr=1e-3)

# MNIST input data (simulated)
X = torch.randn(32, 784)  # 32 samples of 784 features each (MNIST input)
y = torch.randint(0, 10, (32,))  # Labels

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = nn.CrossEntropyLoss()(outputs, y)
    if isinstance(loss, torch.Tensor) and loss.is_cuda:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
    loss.backward()
    optimizer.step()

# Note: The model's large layers and high learning rate may cause NaN loss during training