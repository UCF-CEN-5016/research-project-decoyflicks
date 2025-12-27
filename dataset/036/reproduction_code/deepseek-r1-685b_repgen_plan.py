import torch

def calculate_delta(rewards, values, t, gamma):
    if t == len(rewards) - 1:
        return rewards[t] - values[t]
    else:
        return rewards[t] + gamma * values[t + 1] - values[t]

def calculate_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = calculate_delta(rewards, values, t, gamma)
        gae = delta + (gamma * lam) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

# Test case
rewards = torch.tensor([1.0, 0.5, 0.2])
values = torch.tensor([0.8, 0.6, 0.3])

incorrect = calculate_gae(rewards, values, gamma=0.99, lam=0.95)
correct = calculate_gae(rewards, values, gamma=0.99, lam=0.95)

print(f"Incorrect GAE: {incorrect}")
print(f"Correct GAE: {correct}")
print(f"Difference: {torch.norm(incorrect - correct)}")