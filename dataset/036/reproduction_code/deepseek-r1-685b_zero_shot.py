import numpy as np

def calculate_gae_bugged(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_non_terminal = 1.0
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages.insert(0, gae)
    return np.array(advantages)

def calculate_gae_corrected(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_non_terminal = 1.0
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages.insert(0, gae)
    return np.array(advantages)

rewards = np.array([1.0, 2.0, 3.0])
values = np.array([0.5, 1.0, 1.5])
dones = np.array([0, 0, 0])
gamma = 0.9
lam = 0.8

bugged = calculate_gae_bugged(rewards, values, dones, gamma, lam)
corrected = calculate_gae_corrected(rewards, values, dones, gamma, lam)
print("Bugged:", bugged)
print("Corrected:", corrected)