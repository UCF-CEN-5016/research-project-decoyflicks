

import numpy as np

gamma = 0.9
lambda_ = 0.95
rewards = [0, 1, 2]  # Assuming t=0, t+1=1, t+2=2

# Simulate the bug in GAE calculation
term = gamma**2 * rewards[0 + 1]  # Incorrectly using t+1 instead of t+2
wk = (1 - lambda_) * lambda_**1  # Incorrectly using lambda^1 instead of lambda^(k-1)

print("Incorrect term:", term)
print("Incorrect weight:", wk)