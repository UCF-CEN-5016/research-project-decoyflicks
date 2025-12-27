import numpy as np
import tensorflow as tf

class RandAugment:
    def __init__(self, num_layers=2, magnitude=10, level_std=0.5):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.level_std = level_std

    def apply(self):
        # Incorrect implementation without level_std
        std_dev = np.random.normal(0, 1)
        std_dev = np.clip(std_dev, 0, 1)

        # Correct implementation with level_std
        # std_dev = np.random.normal(0, self.level_std)

        return std_dev

# Minimal setup
np.random.seed(0)
rand_augment = RandAugment()

# Triggering conditions
std_dev = rand_augment.apply()
print("Standard Deviation:", std_dev)

# Verify the bug
if std_dev == 0 or std_dev == 1:
    print("Bug reproduced: Standard deviation is either 0 or 1")
else:
    print("Bug not reproduced: Standard deviation is not 0 or 1")