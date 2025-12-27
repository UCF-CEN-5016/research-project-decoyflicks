import torch

# Define parameters
gamma = 0.99
lambda_ = 0.95
rewards = torch.tensor([1.0, 2.0, 3.0])  # Rewards for t = 0, 1, 2

# Calculate terms and weights correctly
terms = []
for t in range(len(rewards)):
    term = gamma**2 * rewards[t + 1] if t + 1 < len(rewards) else 0.0
    weight = (1 - lambda_) * (lambda_ ** t)
    terms.append((term, weight))

print("Correct terms and weights:")
for term, weight in terms:
    print(f"Term: {term.item()}, Weight: {weight.item()}")