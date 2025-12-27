import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture similar to the provided example but adjusted for immediate perfect fit.
class MinimalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))

# Initialize the model with small dimensions to allow for immediate fit.
model = MinimalModel(10, 49, 10)  # Input and output dimension as per their ViT

# Create a simple dataset where each label matches an input vector exactly
X = torch.randn(32, 10)
y = X[:32]  # Each sample's label is the same as its input features

# Define loss function and optimizer with high learning rate to facilitate quick convergence
 criterion = nn.MSELoss()
 optimizer = optim.SGD(model.parameters(), lr=1.0)

# Training loop with a single epoch to demonstrate immediate fit
for epoch in range(1):  # Single epoch for demonstration
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()  # Zero gradients before backpropagation
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights
    
    print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')