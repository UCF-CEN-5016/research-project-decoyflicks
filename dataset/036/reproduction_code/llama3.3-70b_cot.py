import numpy as np

def calculate_gae(rewards, values, gamma, lambda_):
    """
    Calculate the generalized advantage estimator (GAE) with the bug.
    """
    # Incorrect application of gamma and lambda
    gae = np.zeros(len(rewards))
    for t in range(len(rewards) - 1):
        # Buggy formula: should use gamma^2 * r_{t+2} and correct wk calculation
        gae[t] = rewards[t] + gamma * values[t+1] - values[t]
        for k in range(1, len(rewards) - t):
            # Incorrect application of gamma and lambda
            gae[t] += (gamma ** k) * rewards[t+k]
            # Incorrect calculation of wk
            # Correct formula should be (1-lambda) * lambda^(k-1)
            gae[t] += (gamma ** k) * lambda_ ** (k-1) * rewards[t+k]
    return gae

def correct_calculate_gae(rewards, values, gamma, lambda_):
    """
    Calculate the generalized advantage estimator (GAE) correctly.
    """
    gae = np.zeros(len(rewards))
    for t in range(len(rewards) - 1):
        gae[t] = rewards[t] + gamma * values[t+1] - values[t]
        for k in range(1, len(rewards) - t):
            # Correct application of gamma and lambda
            gae[t] += (gamma ** 2) * rewards[t+k]  # Corrected gamma application
            # Correct calculation of wk
            gae[t] += (1 - lambda_) * (lambda_ ** (k-1)) * rewards[t+k]
    return gae

# Example usage
if __name__ == "__main__":
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    gamma = 0.9
    lambda_ = 0.95
    
    buggy_gae = calculate_gae(rewards, values, gamma, lambda_)
    correct_gae = correct_calculate_gae(rewards, values, gamma, lambda_)
    
    print("Buggy GAE:", buggy_gae)
    print("Correct GAE:", correct_gae)