

import numpy as np

# Define parameters
gamma = 0.9
lambda_ = 0.9

# Rewards for t=0, t+1, t+2
rewards = np.array([10, 20, 30])

# Simulate the incorrect line 36
# Original: gamma^2 * rewards[t+2]
# Incorrect: gamma^2 * rewards[t+1]
incorrect_term = gamma**2 * rewards[0 + 1]  # t=0, using t+1
print("Incorrect term (line 36):", incorrect_term)

# Simulate the incorrect line 45
# Original: (1 - lambda) * lambda^(k-1)
# Incorrect: (1 - lambda) * lambda^k
k = 1  # Assuming k=1
incorrect_wk = (1 - lambda_) * (lambda_)**k
print("Incorrect weight (line 45):", incorrect_wk)