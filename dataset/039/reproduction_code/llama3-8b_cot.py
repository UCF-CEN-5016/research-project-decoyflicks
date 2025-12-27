import numpy as np

class ROPECalculator:
    def __init__(self):
        self.cos_cached = np.random.rand(3, 4, 5)
        self.sin_cached = np.random.rand(3, 4, 5)

    def calculate_x_rope(self, x):
        neg_half_x = -0.5 * x
        x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

# Trigger the bug
rope_calculator = ROPECalculator()
x = np.random.rand(4, 5)
try:
    rope_calculator.calculate_x_rope(x)
except RuntimeError as e:
    print(f"Error: {e}")