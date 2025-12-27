import numpy as np

class CircularShifter:
    """Utility to perform circular shifts on numpy arrays."""
    def __init__(self, shift: int = 2):
        self.shift = shift

    def rotate(self, array: np.ndarray) -> np.ndarray:
        """Return the array circularly shifted by the configured amount."""
        return np.roll(array, self.shift)

def main():
    shifter = CircularShifter()
    data = np.array([1, 2, 3])
    print(shifter.rotate(data))
    print(np.roll(data, 1))

if __name__ == "__main__":
    main()