import torch
import torch.nn as nn
from vit_pytorch import NaViT_Nest_Tensor_3D

# Set up minimal environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a navit_nest_3d model instance
model = NaViT_Nest_Tensor_3D(
    num_classes=10,
    num_blocks=2,
    num_heads=4,
    embed_dim=128,
    patch_size=16,
    dropout=0.1,
)

# Move the model to the device
model.to(device)

# Generate some random input data
input_data = torch.randn(5, 3, 256, 256, 256).to(device)

# Define a simple loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Trigger the bug by attempting to train the model
def train(model, input_data):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, torch.randint(0, 10, (5,)).to(device))

    # Backward pass
    loss.backward()

    # Update the model parameters
    optimizer.step()

try:
    train(model, input_data)
except Exception as e:
    print(f"Error: {e}")