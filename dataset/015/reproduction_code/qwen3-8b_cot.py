import matplotlib.pyplot as plt
import numpy as np

# Sample data: a list of images (numpy arrays)
sample_images = [np.random.rand(256, 256, 3) for _ in range(10)]

def show_batch(images, num_of_examples):
    """
    A simplified version of the problematic function.
    The parameter `num_of_examples` is never used.
    """
    # Display the first image (regardless of num_of_examples)
    plt.figure(figsize=(5, 5))
    plt.imshow(images[0])
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

# Triggering the bug: calling show_batch with different num_of_examples
show_batch(sample_images, num_of_examples=5)  # No effect
show_batch(sample_images, num_of_examples=10) # No effect