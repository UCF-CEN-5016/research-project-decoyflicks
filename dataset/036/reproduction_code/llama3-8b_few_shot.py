import torch

# Define parameters and discount factor
gamma = 0.99
lambda_ = 0.95
rewards = [1, 2, 3]  # Example rewards sequence
values = [0.5, 0.7, 0.9]  # Example values sequence

# Calculate GAE using incorrect formula
gae_rewards = []
for t in range(len(rewards) - 1):
    gae_reward = gamma**2 * rewards[t+1]
    gae_rewards.append(gae_reward)

for t in range(len(values) - 1):
    wk = (1-lambda_) * (lambda_ ** (t-1))
    gae_value = wk * (values[t+1] - values[t])
    print(f"GAE value for step {t}: {gae_value}")