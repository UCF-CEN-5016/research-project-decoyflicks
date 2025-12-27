import matplotlib.pyplot as plt
import numpy as np

def show_batch(batch):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(np.random.rand(100, 100))  # Dummy image
    plt.show()

# Example usage
show_batch(np.random.rand(10, 100, 100))