import numpy as np

# Define parameters
gamma = 0.9
lambda_ = 0.9

# Rewards for t=0, t+1, t+2
rewards = np.array([10, 20, 30])

# Corrected term calculation
correct_term = gamma**2 * rewards[2]  # t=0, using t+2
print("Corrected term (line 36):", correct_term)

# Corrected weight calculation
k = 1  # Assuming k=1
correct_wk = (1 - lambda_) * (lambda_)**(k-1)
print("Corrected weight (line 45):", correct_wk)