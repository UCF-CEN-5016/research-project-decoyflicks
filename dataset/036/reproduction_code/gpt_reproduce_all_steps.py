import numpy as np
from labml_nn.rl.ppo.gae import GAE

n_workers = 1
worker_steps = 4
gamma = 0.9
lambda_ = 0.8

gae = GAE(n_workers, worker_steps, gamma, lambda_)

done = np.zeros((n_workers, worker_steps), dtype=np.float32)
rewards = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
values = np.array([[0.5, 1.0, 1.5, 2.0, 0.0]], dtype=np.float32)

advantages = gae(done, rewards, values)

print("Advantages:", advantages)

# Manual expected calculation for first time step for comparison:
# Correct should include gamma^2 * r_{t+2} = gamma^2 * r_2 = 0.9^2 * 3.0 = 0.81 * 3 = 2.43
# But bug uses gamma^2 * r_{t+1} = 0.81 * 2 = 1.62 (incorrect)
#
# So difference in advantage for t=0 should be visible.
expected_advantage_t0 = (
    rewards[0, 0]
    + gamma * rewards[0, 1]
    + (gamma ** 2) * rewards[0, 2]
    + (gamma ** 3) * rewards[0, 3]
    - values[0, 0]
)
print("Expected advantage t=0 (correct formula):", expected_advantage_t0)
print("Computed advantage t=0:", advantages[0, 0])

# Since weights wk are internal and not exposed by GAE class,
# we can reimplement the weight calculation here to compare with the bug:
k_vals = np.arange(1, worker_steps + 1)
correct_wk = (1 - lambda_) * (lambda_ ** (k_vals - 1))
incorrect_wk = (1 - lambda_) * (lambda_ ** k_vals)  # as per bug report line 45

print("Correct weights (wk):", correct_wk)
print("Incorrect weights (wk):", incorrect_wk)