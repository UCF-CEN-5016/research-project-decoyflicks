import torch
import numpy as np

def incorrect_gae_calc(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Reproduction of the buggy GAE calculation"""
    advantages = []
    advantage = 0
    next_value = 0
    
    # Reverse through time steps
    for r, v in zip(rewards[::-1], values[::-1]):
        delta = r + gamma * next_value - v
        advantage = delta + gamma * gae_lambda * advantage
        advantages.insert(0, advantage)
        next_value = v
    return torch.tensor(advantages)

def correct_gae_calc(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Correct GAE calculation"""
    advantages = []
    advantage = 0
    next_value = 0
    
    # Reverse through time steps
    for i, (r, v) in enumerate(zip(rewards[::-1], values[::-1])):
        delta = r + gamma * next_value - v
        # Correct weight calculation: (1-lambda) * lambda^(k-1)
        weight = (1 - gae_lambda) * (gae_lambda ** i)
        advantage = delta + gamma * gae_lambda * advantage * weight
        advantages.insert(0, advantage)
        next_value = v
    return torch.tensor(advantages)

# Test case - simple trajectory
rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
values = torch.tensor([0.5, 1.0, 1.5, 2.0])

# Calculate both versions
incorrect = incorrect_gae_calc(rewards, values)
correct = correct_gae_calc(rewards, values)

print("Incorrect GAE:", incorrect)
print("Correct GAE:", correct)
print("Difference:", correct - incorrect)