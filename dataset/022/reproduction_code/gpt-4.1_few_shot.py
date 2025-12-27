import numpy as np

class RandAugmentBuggy:
    def __init__(self, magnitude=10, magnitude_std=0.5):
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std

    def get_magnitude(self):
        # Bug: missing multiplication by self.magnitude_std, so std is always 1
        noisy_magnitude = self.magnitude + np.random.randn()  # Should be: np.random.randn() * self.magnitude_std
        return noisy_magnitude

# Instantiate with std=0.5
augment = RandAugmentBuggy(magnitude=10, magnitude_std=0.5)

# Collect magnitudes over many samples
magnitudes = [augment.get_magnitude() for _ in range(10000)]

print(f"Mean magnitude: {np.mean(magnitudes):.3f}")
print(f"Std deviation of magnitude: {np.std(magnitudes):.3f}")