import numpy as np

# Minimal setup
class PPO:
    def __init__(self):
        self.gamma = 0.99
        self.lambda_ = 0.95

ppo = PPO()

def gae_formula(rewards, values, next_values):
    # Calculate GAE values
    gae_values = []
    for i in range(len(rewards) - 1):
        gae_value = (1 - ppo.gamma) * rewards[i] + ppo.gamma * (rewards[i+1] * (ppo.lambda_ ** (i+1)))
        gae_values.append(gae_value)
    return gae_values

# Triggering conditions: Sample data and apply GAE formula
rewards = np.array([0.5, 0.7, 0.3, 0.9])
values = np.array([1.2, 0.8, 1.4, 1.1])
next_values = np.array([1.5, 1.7, 1.2])

gae_values = gae_formula(rewards, values, next_values)

print(gae_values)