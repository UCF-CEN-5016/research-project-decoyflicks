import numpy as np

def calculate_gae(rewards, values, gamma, lambda_):
    gae = np.zeros_like(rewards)
    for t in range(len(rewards) - 1):
        gae[t] = rewards[t] + gamma * values[t + 1] - values[t]
    gae = np.cumsum(gae[::-1])[::-1]
    return gae

def calculate_gae_with_lambda(rewards, values, gamma, lambda_):
    gae = np.zeros_like(rewards)
    for t in range(len(rewards) - 1):
        gae[t] = rewards[t] + gamma * values[t + 1] - values[t]
    for k in range(1, len(rewards)):
        wk = lambda_ ** k
        gae[k - 1] = wk * gae[k - 1]
    return gae

rewards = np.array([1, 2, 3, 4, 5])
values = np.array([0, 0, 0, 0, 0])
gamma = 0.9
lambda_ = 0.9

print(calculate_gae(rewards, values, gamma, lambda_))
print(calculate_gae_with_lambda(rewards, values, gamma, lambda_))