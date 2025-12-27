import torch
from spatial_transformer import STN

# Define the STN model
model = STN()

# Load MNIST dataset
import torchvision
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Train the model with high learning rate (not recommended!)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    for X, y in trainloader:
        optimizer.zero_grad()
        outputs = model(X.view(-1, 784))
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")

print("Final model state:")
print(model.state_dict())