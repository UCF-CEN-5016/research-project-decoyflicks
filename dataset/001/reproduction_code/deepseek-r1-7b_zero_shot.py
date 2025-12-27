import torch
import torch.nn as nn
from torch import optim
import numpy as np
from spatial_transformer_network import SpatialTransformerNetwork

# Small fix for MNIST example with spatial transformer network
def train_spatial_transformer Network():
    # Use small grid size to reduce computations and memory usage
    grid_size = 4
    
    # Create model
    model = SpatialTransformerNetwork(grid_size)
    
    # Use smaller learning rate and gradient clipping to prevent NaN loss
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with gradient clipping
    for epoch in range(10):
        for step, (x, y) in enumerate(mnist_dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            loss = criterion(output, y)
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_value_(model.parameters(), 10)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f'Epoch [{epoch+1}/10], Step [{step}], Loss: {loss.item():.4f}')
    
    print("Training completed successfully")

# Initialize MNIST dataset and dataloader
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST(root='./data', train=False, transform=mnist_transform)

batch_size = 64
num_workers = 2

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Train the model
train_spatial_transformer_network()