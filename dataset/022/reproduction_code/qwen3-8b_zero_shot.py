import numpy as np

def apply_rand_augment(magnitude, num_layers):
    for _ in range(num_layers):
        magnitude += np.random.normal(0, 1)  # Missing level_std multiplication
    return magnitude

# Test the bug
np.random.seed(42)
num_layers = 2
magnitudes = [apply_rand_augment(0, num_layers) for _ in range(1000)]
std = np.std(magnitudes)
print(std)