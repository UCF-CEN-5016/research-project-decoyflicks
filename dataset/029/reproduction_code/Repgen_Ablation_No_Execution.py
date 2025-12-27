import torch
import torchvision.models as models
from torch.optim import Adam

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True).to(device)

# Dummy input tensor
dummy_input = torch.randn(4, 3, 256, 256).to(device)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(2):
    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_input)
    target = dummy_input.clone().detach().to(device)  # Dummy target tensor
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')