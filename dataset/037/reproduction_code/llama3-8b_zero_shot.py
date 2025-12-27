import numpy as np
from scipy.stats import norm

class ValuePE:
    def __init__(self):
        self.rotations = 2

    def forward(self, x):
        return np.roll(x, self.rotations)

value_pe = ValuePE()
x = np.array([1, 2, 3])
print(value_pe.forward(x))  # Output: [3 1 2]
print(np.roll(x, 1))  # Output: [2 3 1]