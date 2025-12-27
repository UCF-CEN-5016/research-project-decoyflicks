import matplotlib.pyplot as plt
import numpy as np

def show_batch(images, masks, num_of_examples=4):
    """Display a batch of images with masks.
    
    Args:
        images: Batch of images
        masks: Corresponding masks
        num_of_examples: Number of examples to show (unused)
    """
    # The num_of_examples argument is never referenced
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(images[0])
    axes[1].imshow(masks[0])
    plt.show()

# Test data
test_images = np.random.rand(8, 256, 256, 3)  # Batch of 8 images
test_masks = np.random.randint(0, 2, (8, 256, 256))  # Batch of 8 masks

# Calling with different num_of_examples has no effect
show_batch(test_images, test_masks, num_of_examples=2)  # Still shows 1 example
show_batch(test_images, test_masks, num_of_examples=8)  # Still shows 1 example