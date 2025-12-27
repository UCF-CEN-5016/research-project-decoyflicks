import torch
from torch import nn
import torch.nn.functional as F

class GAE(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, lambda_=0.95):
        super(GAE, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.gamma = gamma
        self.lambda_ = lambda_

    def forward(self, rewards):
        # Incorrect application of gamma squared on rewards_{t+1}
        gamma_squared = self.gamma ** 2
        advantage = [gamma_squared * r for r in rewards[1:]]  # Assuming rewards is a list/tensor
        
        # Compute correct weights with exponent k-1
        k = torch.arange(1, len(rewards)) + 1  # Assuming len(rewards) corresponds to timesteps
        wk = (1 - self.lambda_) * (self.lambda_ ** (k - 1))
        
        return advantage, wk

# Example usage:
state_dim = 4  # Example state dimension
action_dim = 2  # Example action dimension
model = GAE(state_dim, action_dim)

# Generate dummy rewards for t+1 steps
rewards = torch.randn(5)  # 5 rewards: t=0 to t=4

# Incorrectly applied gamma squared on r_{t+1}
advantages_incorrect = model(rewards)[0]
weights_incorrect = model.rewards[1:]  # Note: This may not be correct as per the bug details

print("Advantages with incorrect gamma^2 application:", advantages_incorrect)
print("Weights using exponent k-1:", weights_incorrect)