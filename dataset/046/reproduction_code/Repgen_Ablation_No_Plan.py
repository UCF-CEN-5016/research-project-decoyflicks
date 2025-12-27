import numpy as np
from scipy.special import softmax

def calculate_log_probs(data):
    data_prob = softmax(data, axis=2)
    return data_prob

# Example usage
data = np.random.rand(5, 10, 3)  # Replace with actual data
log_probs = calculate_log_probs(data)

print(log_probs.shape)