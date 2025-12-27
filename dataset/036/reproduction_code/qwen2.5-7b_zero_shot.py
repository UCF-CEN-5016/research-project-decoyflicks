import numpy as np

gamma = 0.9
lambda_ = 0.95
rewards = [0, 1, 2]  # Assuming t=0, t+1=1, t+2=2

# Corrected calculation of GAE
term = gamma**2 * rewards[0 + 2]  # Using t+2 instead of t+1
wk = (1 - lambda_) * lambda_**(2 - 1)  # Using lambda^(k-1) instead of lambda^1

print("Corrected term:", term)
print("Corrected weight:", wk)