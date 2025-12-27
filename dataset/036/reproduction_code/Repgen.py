import torch

# Define a batch size of 1 and gamma value of 0.95 for reproducibility
batch_size = 1
gamma = 0.95

# Create a synthetic environment with 2 states, each having a reward of [0, -1] and next state indices [1, 0]
states = [[0, -1], [0, -1]]
next_state_indices = [[1, 0], [1, 0]]

# Initialize the GAE formula with lambda=0.95, gamma^2 r_{t+1} and wk calculation as (1-lambda) * lambda ^ (k-1)
lambda_value = 0.95
gamma_squared_r_t1 = gamma ** 2 * torch.tensor([-1, -1])
wk = (1 - lambda_value) * lambda_value ** (batch_size - 1)

# Run the environment for 10 steps to observe the bug
for step in range(10):
    # Calculate returns using the GAE formula
    returns = gamma_squared_r_t1 + wk

    # Record the calculated returns for verification
    print(f'Step {step}: Returns = {returns.item()}')

# Verify that the calculated returns do not match the expected values due to incorrect formula usage
expected_returns = torch.tensor([0.95 ** 2 * -1, 0.95 ** 2 * -1])
print("Calculated Returns:", returns)
print("Expected Returns:", expected_returns)