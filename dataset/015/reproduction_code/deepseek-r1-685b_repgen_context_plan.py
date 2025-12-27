import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def show_batch(images, labels, display_count=9):
    """
    Displays a batch of images with their corresponding labels in a 3x3 grid.

    Args:
        images (np.ndarray): A NumPy array of images. Expected shape (N, H, W, C).
        labels (list): A list of labels corresponding to the images.
        display_count (int): The maximum number of images to display.
                             Defaults to 9, constrained by the 3x3 grid.
    """
    if not isinstance(images, np.ndarray) or images.ndim != 4:
        print("Error: images must be a 4D NumPy array (N, H, W, C).")
        return
    if not isinstance(labels, list):
        print("Error: labels must be a list.")
        return

    num_to_show = min(display_count, len(images), len(labels), 9) # Limit to 9 for 3x3 grid

    if num_to_show == 0:
        print("No images to display.")
        return

    plt.figure(figsize=(10, 10))
    for i in range(num_to_show):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype('uint8'))
        plt.title(labels[i])
        plt.axis('off')
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show()

if __name__ == "__main__":
    # Sample data
    sample_images = np.random.randint(0, 256, size=(10, 256, 256, 3), dtype=np.uint8)
    sample_labels = [f'label{i+1}' for i in range(10)]

    # Call the function with the display_count argument
    print("Displaying 5 images:")
    show_batch(sample_images, sample_labels, 5)

    print("\nDisplaying 9 images (default behavior):")
    show_batch(sample_images, sample_labels)

    print("\nAttempting to display more than available images:")
    show_batch(sample_images[:3], sample_labels[:3], 5)

    print("\nAttempting to display with insufficient data:")
    show_batch(np.array([]).reshape(0,256,256,3), [], 5)
