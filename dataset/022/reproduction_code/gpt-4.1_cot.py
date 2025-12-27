import tensorflow as tf
import numpy as np

# Simulate parameters
level_std = 0.5  # Intended standard deviation scale for noise
magnitude = tf.constant(5.0)

# Buggy code: noise added without scaling by level_std
def buggy_augment(magnitude, level_std):
    # Noise stddev is always 1 here
    magnitude_noisy = magnitude + tf.random.normal(shape=[], mean=0.0, stddev=1.0)
    return magnitude_noisy

# Correct code: noise scaled by level_std
def fixed_augment(magnitude, level_std):
    magnitude_noisy = magnitude + level_std * tf.random.normal(shape=[], mean=0.0, stddev=1.0)
    return magnitude_noisy

# Run multiple samples to check std deviation of noise
samples = 10000

buggy_samples = [buggy_augment(magnitude, level_std).numpy() for _ in range(samples)]
fixed_samples = [fixed_augment(magnitude, level_std).numpy() for _ in range(samples)]

buggy_noise = np.array(buggy_samples) - magnitude.numpy()
fixed_noise = np.array(fixed_samples) - magnitude.numpy()

print("Buggy noise stddev:", np.std(buggy_noise))  # Expected ~1.0 regardless of level_std
print("Fixed noise stddev:", np.std(fixed_noise))  # Expected ~level_std (0.5)

# Confirming the bug that the added noise stddev is always 1 in buggy case