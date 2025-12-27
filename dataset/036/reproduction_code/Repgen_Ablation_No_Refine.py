import torch
from torch import nn

gamma = 0.99
lambda_ = 0.95

# Create random input tensor for demonstration purposes
x = torch.randn(1, 4, 64)

# Calculate r_t+2 as a random scalar value
r_t2 = torch.tensor(0.8)

# Calculate gamma^2 * r_{t+2}
gamma_squared_r_t2 = gamma**2 * r_t2

# Set k to 3 for demonstration purposes
k = 3

# Calculate wk using the formula (1-lambda) * lambda ^ (k-1)
wk = (1 - lambda_) * lambda_**(k-1)

# Assert that wk is equal to 0.0240625
assert torch.isclose(wk, torch.tensor(0.0240625))

# Calculate advantages using the modified gamma^2 r_{t+2} and wk for demonstration purposes
advantages = gamma_squared_r_t2 * wk

# Verify that the advantages tensor contains incorrect values due to the bug
print(advantages)

# Set up a loop to iterate through multiple episodes, each time demonstrating the bug
for episode in range(5):
    print(f"Episode {episode+1}:")
    print(advantages)