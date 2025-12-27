

import matplotlib.pyplot as plt
import numpy as np

def show_batch(batch, num_of_examples):
    # Unused parameter num_of_examples
    plt.figure(figsize=(10, 5))
    for i in range(5):  # Hardcoded to 5, ignoring num_of_examples
        plt.subplot(1, 5, i+1)
        plt.imshow(np.random.rand(100, 100))  # Dummy image
    plt.show()

# Example usage
show_batch(np.random.rand(10, 100, 100), num_of_examples=3)

import matplotlib.pyplot as plt
import numpy as np

def show_batch(batch, num_of_examples):
    # Unused parameter num_of_examples
    plt.figure(figsize=(10, 5))
    for i in range(5):  # Hardcoded to 5, ignoring num_of_examples
        plt.subplot(1, 5, i+1)
        plt.imshow(np.random.rand(100, 100))  # Dummy image
    plt.show()

# Example usage
show_batch(np.random.rand(10, 100, 100), num_of_examples=3)