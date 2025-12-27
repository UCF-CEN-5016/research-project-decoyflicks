import torch
from vit_pytorch import navit_nest_3d

# Load the model
model = navit_nest_3d()

# Sample data (assuming this is where the issue lies)
X = torch.randn(32, 10)

# Train the model
for epoch in range(1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    outputs = model(X)
    loss = torch.nn.functional.mse_loss(outputs, X)  # Assuming this is where the error occurs
    loss.backward()