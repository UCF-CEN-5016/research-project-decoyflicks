import torch

def incorrect_gae(rewards, values, gamma=0.99, lam=0.95):
    """Demonstrates the incorrect GAE implementation"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            # Bug 1: Incorrect gamma exponent (should be t+2 not t+1)
            delta = rewards[t] + gamma * values[t+1] - values[t]
        
        # Bug 2: Incorrect lambda exponent (should be k-1)
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

def correct_gae(rewards, values, gamma=0.99, lam=0.95):
    """Correct GAE implementation for comparison"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t+1] - values[t]
        
        # Correct lambda weighting
        gae = delta + (gamma * lam) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

# Test case
rewards = torch.tensor([1.0, 0.5, 0.2])
values = torch.tensor([0.8, 0.6, 0.3])

incorrect = incorrect_gae(rewards, values)
correct = correct_gae(rewards, values)

print(f"Incorrect GAE: {incorrect}")
print(f"Correct GAE: {correct}")
print(f"Difference: {torch.norm(incorrect - correct)}")