import torch

def calculate_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t+1] - values[t]
        
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

# Test case
rewards = torch.tensor([1.0, 0.5, 0.2])
values = torch.tensor([0.8, 0.6, 0.3])

advantages = calculate_gae(rewards, values)
print(f"Advantages: {advantages}")