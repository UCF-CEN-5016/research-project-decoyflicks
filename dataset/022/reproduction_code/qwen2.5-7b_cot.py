import numpy as np

def generate_magnitude(level_std):
    """
    Simulates the RandAugment magnitude calculation
    where the standard deviation is based on the provided level_std.
    """
    magnitude = np.random.normal(0, level_std)
    return magnitude

def calculate_std_dev(magnitudes):
    """
    Calculates the standard deviation of a list of magnitudes.
    """
    return np.std(magnitudes)

# Test with a level_std that should influence the standard deviation
level_std = 0.5
magnitudes = [generate_magnitude(level_std) for _ in range(1000)]
std_dev = calculate_std_dev(magnitudes)

print(f"Standard deviation of magnitudes: {std_dev}")
print(f"Expected standard deviation (based on level_std): {level_std}")