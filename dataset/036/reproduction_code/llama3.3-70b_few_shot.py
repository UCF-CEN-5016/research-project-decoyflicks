import numpy as np

def calculate_gae(rewards, values, gamma, lambda_):
    # Incorrect implementation
    gae = np.zeros_like(rewards)
    for t in range(len(rewards) - 1):
        gae[t] = rewards[t] + gamma * values[t + 1] - values[t]
        gae[t] += gamma ** 2 * rewards[t + 1]  # Should be gamma^2 * r_{t+2}
    
    # Incorrect calculation of weights
    weights = np.zeros(len(rewards))
    for k in range(1, len(rewards)):
        weights[k] = lambda_ ** (k - 1)  # Should be (1-lambda) * lambda ^ (k-1)
    
    return gae, weights

# Sample data
rewards = np.array([1, 2, 3, 4, 5])
values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
gamma = 0.9
lambda_ = 0.95

# Calculate GAE with incorrect formula
gae, weights = calculate_gae(rewards, values, gamma, lambda_)

print("GAE:", gae)
print("Weights:", weights)