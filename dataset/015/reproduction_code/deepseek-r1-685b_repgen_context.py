import matplotlib.pyplot as plt
import numpy as np

def show_batch(images, masks):
    """Display a batch of images with masks.

    Args:
        images: Batch of images
        masks: Corresponding masks
    """
    num_of_examples = len(images)
    fig, axes = plt.subplots(1, num_of_examples, figsize=(5*num_of_examples, 5))
    for i in range(num_of_examples):
        axes[i].imshow(images[i])
        axes[i].set_title(f'Example {i+1}')
        axes[i].axis('off')
    plt.show()

# Test data
test_images = np.random.rand(4, 256, 256, 3)  # Batch of 4 images
test_masks = np.random.randint(0, 2, (4, 256, 256))  # Batch of 4 masks

show_batch(test_images, test_masks)