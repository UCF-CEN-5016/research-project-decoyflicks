import numpy as np

def generate_magnitude(level_std):
    """
    Simulates the buggy RandAugment magnitude calculation
    where the standard deviation is fixed at 1 instead of using level_std.
    """
    # Buggy code: standard deviation is fixed at 1
    magnitude = np.random.normal(0, 1)
    return magnitude

# Test with a level_std that should influence the standard deviation
level_std = 0.5  # This parameter has no effect in the buggy implementation
magnitudes = [generate_magnitude(level_std) for _ in range(1000)]
std_dev = np.std(magnitudes)

print(f"Standard deviation of magnitudes: {std_dev}")
print(f"Expected standard deviation (based on level_std): {level_std}")