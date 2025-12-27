import matplotlib.pyplot as plt
import numpy as np

# Sample data: a list of images (numpy arrays)
sample_images = [np.random.rand(256, 256, 3) for _ in range(10)]

def show_image(image):
    """
    Display a single image.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

def show_first_image(images):
    """
    Display the first image from a list of images.
    """
    show_image(images[0])

# Display the first image
show_first_image(sample_images)