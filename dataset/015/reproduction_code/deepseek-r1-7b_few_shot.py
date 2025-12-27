import matplotlib.pyplot as plt
import numpy as np

# Creating dummy data to simulate an image batch
def create_dummy_batch(num_images=32, height=256, width=256):
    """Generates random images and labels for demonstration."""
    # Random RGB images with random class labels
    images = np.random.randn(num_images, height, width, 3)
    labels = np.random.randint(0, 100, size=(num_images,))
    
    return {'images': images}, labels

# The function that demonstrates the issue
def show_batch(images, num_of_examples=32, figure_size=(10, 6)):
    """Shows a batch of images."""
    plt.figure(figsize=figure_size)
    for i in range(num_of_examples):
        plt.imshow(images[i])
        plt.colorbar()
        plt.title(f"Image {i+1}")
        plt.show()

# Creating sample data
batch = create_dummy_batch(32)

# Demonstrating the function with an unused argument
show_batch(batch['images'], num_of_examples=16, figure_size=(20, 8))