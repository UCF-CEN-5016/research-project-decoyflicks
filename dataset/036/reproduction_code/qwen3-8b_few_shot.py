import torch

# Define parameters
gamma = 0.99
lambda_ = 0.95
rewards = torch.tensor([1.0, 2.0, 3.0])  # Rewards for t = 0, 1, 2

# Simulate the incorrect calculation
terms = []
for k in range(1, len(rewards) + 1):
    # First error: using gamma^2 * rewards[k] instead of gamma^2 * rewards[k+1]
    # Second error: using (1 - lambda_) * lambda_ ** k instead of (1 - lambda_) * lambda_ ** (k-1)
    term = gamma**2 * rewards[k]  # Incorrect term
    wk = (1 - lambda_) * (lambda_ ** k)  # Incorrect weight
    terms.append((term, wk))

print("Incorrect terms and weights:")
for term, wk in terms:
    print(f"Term: {term.item()}, Weight: {wk.item()}")

term = gamma**2 * rewards[k + 1] if k + 1 < len(rewards) else 0.0
wk = (1 - lambda_) * (lambda_ ** (k - 1))